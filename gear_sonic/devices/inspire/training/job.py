"""Portable basket pilot preflight/launcher. Default: validate, never optimize.

Requires the existing Isaac-GR00T training environment. This file is copied into
submission bundles so a training server does not need WBC, MuJoCo or drivers.
"""

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def linear_patch_projection(module, hidden_states):
    """Qwen's non-overlapping patch Conv3d, using the same weights as a GEMM.

    Avoids the torch 2.9/cuDNN bf16 Conv3d regression. Parameters, gradients
    and checkpoint keys stay unchanged. This is not a general Conv3d rewrite.
    https://github.com/huggingface/transformers/issues/45750
    """
    import torch.nn.functional as F

    projection = module.proj
    kernel = (module.temporal_patch_size, module.patch_size, module.patch_size)
    if (projection.kernel_size != kernel or projection.stride != kernel
            or projection.padding != (0, 0, 0) or projection.dilation != (1, 1, 1)
            or projection.groups != 1):
        raise ValueError("unsupported Qwen patch projection geometry")
    patches = hidden_states.reshape(-1, module.in_channels * kernel[0] * kernel[1] * kernel[2])
    return F.linear(patches.to(projection.weight.dtype), projection.weight.flatten(1), projection.bias)


def configure_gpu_runtime():
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; CPU training is disabled")
    Qwen3VLVisionPatchEmbed.forward = linear_patch_projection
    print("QWEN_PATCH_PROJECTION=linear (original weights and checkpoint keys retained)", flush=True)
    print(f"CUDA_TRAINING devices={torch.cuda.device_count()} name={torch.cuda.get_device_name(0)}", flush=True)


def data_paths(bundle, manifest, override=None):
    root = (
        Path(override).resolve()
        if override
        else (bundle / "datasets" if (bundle / "datasets").is_dir() else Path(manifest["dataset_root"]))
    )
    result = {name: (root / path).resolve() for name, path in manifest["datasets"].items()}
    if any(not path.is_relative_to(root.resolve()) for path in result.values()):
        raise ValueError("dataset path escapes root")
    return root, result


def verify_files(bundle, manifest, checkpoint, root):
    for name, expected in (manifest["checkpoint_shards_sha256"] | manifest["checkpoint_config_sha256"]).items():
        if digest(checkpoint / name) != expected:
            raise ValueError("checkpoint hash differs: " + name)
    for name in ("config.json", "processor_config.json", "statistics.json", "embodiment_id.json"):
        if digest(bundle / name) != manifest["checkpoint_config_sha256"][name]:
            raise ValueError("bundle checkpoint config differs: " + name)
    for name, expected in manifest["dataset_files_sha256"].items():
        # Stats are refreshed by the official GR00T loader, not a source label.
        if Path(name).name in ("stats.json", "relative_stats.json"):
            continue
        if digest(root / name) != expected:
            raise ValueError("dataset hash differs: " + name)


def record_runtime(groot_root):
    def git(*args):
        result = subprocess.run(["git", "-C", str(groot_root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    packages = {}
    for name in (
        "torch",
        "transformers",
        "numpy",
        "pyarrow",
        "av",
        "opencv-python",
        "opencv-python-headless",
        "datasets",
        "huggingface-hub",
        "accelerate",
        "flash-attn",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    source_path = Path(groot_root) / "SOURCE.json"
    source = read(source_path) if source_path.is_file() else {}
    return {
        "python": platform.python_version(),
        "groot_commit": git("rev-parse", "HEAD") or source.get("commit"),
        "groot_tracked_changes": git("status", "--porcelain", "--untracked-files=no"),
        "groot_source_snapshot": source,
        "packages": packages,
        "weights_loaded_locally": True,
        "tokenizer_metadata_may_access_hub": True,
    }


def make_config(bundle, manifest, paths, checkpoint, output, num_gpus):
    from gr00t.configs.base_config import get_default_config
    import yaml

    raw = yaml.safe_load((bundle / "finetune.yaml").read_text())
    # Base N1.7 has no SONIC statistics. Reuse GR00T's post-training modality;
    # the dataset factory estimates normalization from the training split only.
    config = get_default_config()
    raw["data"]["modality_configs"] = {"unitree_g1_sonic": config.data.modality_configs["unitree_g1_sonic"]}
    for dataset in raw["data"]["datasets"]:
        keys = dataset["dataset_paths"]
        if not keys or any(key not in paths or not key.endswith("_train") for key in keys):
            raise ValueError("training datasets must name registered training splits")
        dataset["dataset_paths"] = [str(paths[key]) for key in keys]
    training = raw["training"]
    old_gpus = int(training["num_gpus"])
    old_batch = int(training["global_batch_size"])
    effective_batch = old_batch * int(training["gradient_accumulation_steps"])
    if old_batch % old_gpus:
        raise ValueError("configured batch must divide evenly across GPUs")
    new_batch = old_batch // old_gpus * num_gpus
    if effective_batch % new_batch:
        raise ValueError("GPU count cannot preserve the configured effective batch")
    training.update(
        start_from_checkpoint=str(checkpoint),
        output_dir=str(output),
        num_gpus=num_gpus,
        global_batch_size=new_batch,
        gradient_accumulation_steps=effective_batch // new_batch,
    )
    config = config.load_dict(raw)
    config.validate()
    return config


def validate_data(bundle, manifest, paths, config):
    import av
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.stats import generate_rel_stats, generate_stats
    from gr00t.data.types import EmbodimentTag
    import numpy as np
    import pyarrow.parquet as pq

    train = set(manifest["source_split"]["train"])
    val = set(manifest["source_split"]["validation"])
    if not train or not val or train & val:
        raise ValueError("source episode leakage")
    details = {}
    modality = config.data.modality_configs["unitree_g1_sonic"]
    for name, path in paths.items():
        if (path / "INCOMPLETE").exists():
            raise ValueError("incomplete dataset")
        info = read(path / "meta/info.json")
        provenance = read(path / "meta/provenance.json")
        if info["fps"] != 50:
            raise ValueError("checkpoint requires 50 Hz")
        generate_stats(str(path))
        generate_rel_stats(str(path), EmbodimentTag.UNITREE_G1_SONIC)
        loader = LeRobotEpisodeLoader(path, modality, video_backend="opencv")
        stats = loader.get_dataset_statistics()
        if [
            len(stats["action"][k]["mean"]) for k in ("motion_token", "left_hand_joints", "right_hand_joints")
        ] != [64, 6, 6]:
            raise ValueError("action dimensions differ from checkpoint")
        offset = 0
        episodes = []
        allowed = train if name.endswith("train") else val
        for i in range(info["total_episodes"]):
            record = provenance["episodes"][i]
            if record["source_episode"] not in allowed:
                raise ValueError("augmentation split leakage")
            t = pq.read_table(path / info["data_path"].format(episode_chunk=i // 1000, episode_index=i))
            n = t.num_rows
            for key in t.column_names:
                values = np.asarray(t[key].to_pylist())
                if not np.isfinite(values).all():
                    raise ValueError(f"nonfinite {name}/{i}/{key}")
                expected = info["features"][key]
                if (
                    values.dtype.kind == "f"
                    and values.shape != (n, *expected["shape"])
                    and expected["shape"] != [1]
                ):
                    raise ValueError(f"bad shape {name}/{i}/{key}")
            assert np.allclose(t["timestamp"].to_numpy(), np.arange(n) / 50, atol=2e-6)
            assert np.array_equal(t["frame_index"].to_numpy(), np.arange(n))
            assert np.array_equal(t["index"].to_numpy(), np.arange(offset, offset + n))
            assert np.all(t["episode_index"].to_numpy() == i)
            assert np.all(t["task_index"].to_numpy() == 0)
            v = path / info["video_path"].format(
                episode_chunk=i // 1000, episode_index=i, video_key="observation.images.ego_view"
            )
            count = 0
            with av.open(str(v)) as container:
                stream = container.streams.video[0]
                assert float(stream.average_rate) == 50
                for frame in container.decode(stream):
                    assert (frame.height, frame.width) == (480, 640)
                    assert abs(float(frame.pts * frame.time_base) - count / 50) < 1e-5
                    count += 1
            assert count == n == record["frames"] and n >= 40
            parsed = loader._load_parquet_data(i)
            assert len(parsed) == n and np.vstack(parsed["action.motion_token"]).shape == (n, 64)
            sampled = loader._load_video_data(i, np.array([0, n // 2, n - 1]))
            assert sampled["ego_view"].shape == (3, 480, 640, 3)
            episodes.append({"episode": i, "frames": n, "decoded_frames": count})
            offset += n
        assert offset == info["total_frames"]
        details[name] = {
            "episodes": episodes,
            "frames": offset,
            "loader": "LeRobotEpisodeLoader",
            "all_video_frames_decoded": True,
        }
        print(f"DATA_CHECK_OK {name} episodes={len(episodes)} frames={offset}", flush=True)
    return details


def check_forward(bundle, paths, config):
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    from gr00t.data.types import EmbodimentTag
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
    import numpy as np
    import torch
    from transformers import set_seed

    set_seed(config.data.seed)
    artifacts = bundle / "preflight"
    artifacts.mkdir(exist_ok=True)
    pipeline = Gr00tN1d7Pipeline(config, artifacts)
    pipeline.setup()
    model = pipeline.return_model().to("cuda")
    model.eval()
    processor = pipeline.return_processor()
    processor.eval()
    collator = pipeline.return_collator()
    details = {}
    try:
        for name, path in paths.items():
            ds = ShardedSingleStepDataset(
                path,
                EmbodimentTag.UNITREE_G1_SONIC,
                config.data.modality_configs["unitree_g1_sonic"],
                video_backend="opencv",
                shard_size=64,
                episode_sampling_rate=1.0,
            )
            ds.set_processor(processor)
            episode = ds.episode_loader[0]
            samples = [ds.get_datapoint(episode, index) for index in (0, len(episode) // 2)]
            batch = collator(samples)
            inputs = batch["inputs"]
            mask = inputs["action_mask"]
            assert list(inputs["action"].shape) == [2, 40, 132]
            assert torch.all(mask[:, :, :76] == 1) and torch.all(mask[:, :, 76:] == 0)
            assert torch.isfinite(inputs["action"]).all() and torch.isfinite(inputs["state"]).all()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(**batch)
            loss = float(result["loss"].detach().cpu())
            if not np.isfinite(loss):
                raise ValueError("nonfinite training loss")
            details[name] = {
                "forward_loss": loss,
                "batch": 2,
                "horizon": 40,
                "action_dim": 76,
                "padded_dim": 132,
                "optimizer_steps": 0,
                "model_parameters_updated": False,
            }
            del episode, batch, samples, result, ds
            print("FORWARD_CHECK_OK " + name + " loss=" + str(loss), flush=True)
        details["peak_cuda_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    finally:
        dataset = pipeline.return_dataset()[0]
        if hasattr(dataset, "close"):
            dataset.close()
    return details


def default_training_output(bundle):
    # Local prepared bundles live in outputs/training/<task>. Portable bundles
    # outside that layout use the working directory's outputs folder.
    bundle = Path(bundle).resolve()
    outputs = (bundle.parent.parent if bundle.parent.name == "training"
               and bundle.parent.parent.name == "outputs" else Path.cwd() / "outputs")
    return outputs / "checkpoints" / "finetuned" / bundle.name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", default=str(Path(__file__).parent))
    p.add_argument("--groot-root", required=True)
    p.add_argument("--checkpoint", help="Model directory on this machine; defaults to original local path")
    p.add_argument("--dataset-root", help="Relocate datasets on the training machine")
    p.add_argument("--output", help="New training output directory; default: outputs/checkpoints/finetuned/<bundle-name>")
    p.add_argument("--num-gpus", type=int, choices=(1, 2, 4), default=1)
    p.add_argument(
        "--execute", action="store_true", help="Actually fine-tune; default only checks data/model forward"
    )
    a = p.parse_args()
    bundle = Path(a.bundle).resolve()
    manifest = read(bundle / "manifest.json")
    sys.path.insert(0, str(Path(a.groot_root).resolve()))
    # Keep checkpoint/backbone weights local via transformers_local_files_only.
    # Transformers 4.57.3 still queries public tokenizer metadata on the Hub;
    # forcing its global offline switch here would break that existing runtime.
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    checkpoint = Path(a.checkpoint or manifest["checkpoint"]).resolve()
    root, paths = data_paths(bundle, manifest, a.dataset_root)
    verify_files(bundle, manifest, checkpoint, root)
    output = Path(a.output).resolve() if a.output else default_training_output(bundle)
    config = make_config(bundle, manifest, paths, checkpoint, output, a.num_gpus)
    if a.execute:
        if (bundle / "DO_NOT_TRAIN").exists() or not manifest.get("physics_grasp_verified"):
            raise ValueError(
                "This batch has no verified physics grasp recordings; it is not released for training"
            )
        if (bundle / "INCOMPLETE").exists() or not (bundle / "validation.json").is_file():
            raise ValueError("run default preflight first")
        if read(bundle / "validation.json")["status"] != "PASS":
            raise ValueError("preflight did not pass")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("choose a new training output directory")
        if int(os.environ.get("WORLD_SIZE", "1")) != a.num_gpus:
            raise ValueError("launch multiple GPUs with torchrun --nproc_per_node=N")
        configure_gpu_runtime()
        from gr00t.experiment.experiment import run

        run(config)
    else:
        if a.num_gpus != 1:
            raise ValueError("preflight uses one GPU; training may use 1/2/4")
        config.save(bundle / "resolved_config.yaml")
        details = validate_data(bundle, manifest, paths, config)
        result = check_forward(bundle, paths, config)
        write(
            bundle / "validation.json",
            {
                "status": "PASS",
                "data": details,
                "forward": result,
                "optimizer_steps": 0,
                "base_model": manifest["base_model"],
                "checkpoint_shards_sha256": manifest["checkpoint_shards_sha256"],
                "training_datasets": [dataset.dataset_paths for dataset in config.data.datasets],
                "physics_grasp_verified": manifest["physics_grasp_verified"],
                "scope": manifest["validation_scope"],
                "runtime": record_runtime(a.groot_root),
            },
        )
        (bundle / "INCOMPLETE").unlink(missing_ok=True)
        print("BASKET_READY " + str(bundle), flush=True)


if __name__ == "__main__":
    main()
