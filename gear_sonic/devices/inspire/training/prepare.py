"""Prepare physics-recorded simulated basket grasps for GR00T N1.7; never trains."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from gear_sonic.utils.hand_control.registry import ROOT, deployment

from ..policy import ACTION_DIMS
from ..recording.schema import lerobot_modality_config


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def base_model_metadata(checkpoint):
    config = read(checkpoint / "config.json")
    if config["model_type"] != "Gr00tN1d7" or config["action_horizon"] != 40:
        raise ValueError("expected GR00T N1.7 with action horizon 40")
    if config["max_action_dim"] < 76 or config["max_state_dim"] < 44:
        raise ValueError("base model cannot represent the Inspire/SONIC dimensions")
    shards = set(read(checkpoint / "model.safetensors.index.json")["weight_map"].values())
    if not shards or any(Path(name).name != name for name in shards):
        raise ValueError("invalid model shard index")
    weights = {name: digest(checkpoint / name) for name in sorted(shards)}
    # Official N1.7 release LFS digests; reject a task-finetuned model here.
    expected = {
        "model-00001-of-00002.safetensors": "8a1a1d8a33c99103c7c80c136073c5bb8bfe9ca8f7a970c93c033ea89742906d",
        "model-00002-of-00002.safetensors": "c3f61940deb2007ba1ad7743013b57f0f8462356151db9655175d7aca2d40661",
    }
    if weights != expected:
        raise ValueError("expected official nvidia/GR00T-N1.7-3B base weights")
    return {
        "base_model": {"repo_id": "nvidia/GR00T-N1.7-3B", "revision": "2fc962b973bccdd5d8ce4f67cc63b264d6886495"},
        "checkpoint": str(checkpoint),
        "checkpoint_shards_sha256": weights,
        "checkpoint_config_sha256": {
            name: digest(checkpoint / name)
            for name in (
                "config.json",
                "processor_config.json",
                "statistics.json",
                "embodiment_id.json",
                "model.safetensors.index.json",
            )
        },
    }


def write_training_files(bundle, checkpoint, seed):
    import yaml

    for name in ("config.json", "processor_config.json", "statistics.json", "embodiment_id.json"):
        shutil.copyfile(checkpoint / name, bundle / name)
    config = {
        "model": {
            "model_name": "nvidia/Cosmos-Reason2-2B",
            "load_bf16": False,
            "reproject_vision": False,
            "backbone_trainable_params_fp32": True,
            "use_relative_action": True,  # SONIC modality declares all three actions absolute.
            "tune_llm": True,
            "tune_visual": True,
            "tune_projector": True,
            "tune_diffusion_model": True,
            "color_jitter_params": None,
            "random_rotation_angle": None,
        },
        "data": {
            "datasets": [{"dataset_paths": ["sim_train"], "embodiment_tag": "unitree_g1_sonic", "mix_ratio": 1.0}],
            "seed": seed,
            "video_backend": "torchcodec",
            "shard_size": 256,
            "episode_sampling_rate": 1.0,
            "num_shards_per_epoch": 200,
            "override_pretraining_statistics": True,
            "allow_padding": False,
        },
        "training": {
            "num_gpus": 1,
            "use_ddp": False,
            "global_batch_size": 64,
            "gradient_accumulation_steps": 4,
            "max_steps": 2000,
            "learning_rate": 2e-5,
            "warmup_ratio": 0.05,
            "weight_decay": 1e-5,
            "optim": "adamw_torch_fused",
            "save_steps": 250,
            "save_total_limit": 8,
            "dataloader_num_workers": 4,
            "use_wandb": False,
            "resume_from_checkpoint": False,
            "save_only_model": False,
            "eval_strategy": "no",
            "transformers_local_files_only": True,
        },
    }
    (bundle / "finetune.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copyfile(Path(__file__).with_name("job.py"), bundle / "train.py")
    shutil.copyfile(Path(__file__).with_name("cloud.slurm"), bundle / "cloud.slurm")


TASK = "Grasp the basket on the chair with both hands and lift it."


def export_split(output, episodes, robot_model, split):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gear_sonic.data.exporter import Gr00tDatasetMetadata, compute_episode_stats
    from ..recording.schema import lerobot_features

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    stage = output.with_name("." + output.name + ".staging")
    modality = lerobot_modality_config(robot_model)
    modality["action"] = {k: modality["action"][k] for k in ACTION_DIMS}
    columns = sorted({v["original_key"] for kind in ("state", "action") for v in modality[kind].values()})
    all_features = lerobot_features(robot_model)
    features = {k: deepcopy(all_features[k]) for k in columns + ["observation.images.ego_view"]}
    meta = Gr00tDatasetMetadata.create(
        modality_config=modality,
        script_config={"mode": "sonic_physics_basket_expert"},
        repo_id="local/basket_grasp_sim25",
        root=stage,
        fps=50,
        features=features,
        use_videos=True,
        robot_type="unitree_g1_inspire_rh56dfx",
    )
    meta.add_task(TASK)
    (stage / "INCOMPLETE").write_text("Exporting physics recordings.\n")
    records = []
    for index, episode in enumerate(episodes):
        sample = np.load(episode / "samples.npz")
        n = len(sample["timestamp"])
        arrays = {
            "observation.state": sample["state"],
            "observation.left_hand": sample["left_hand"],
            "observation.right_hand": sample["right_hand"],
            "observation.projected_gravity": sample["projected_gravity"],
            "action.motion_token": sample["motion_token"],
            "action.left_hand_joints": sample["hand_action"][:, 0],
            "action.right_hand_joints": sample["hand_action"][:, 1],
            "timestamp": np.arange(n, dtype=np.float32) / 50,
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, index, dtype=np.int64),
            "index": np.arange(meta.total_frames, meta.total_frames + n, dtype=np.int64),
            "task_index": np.zeros(n, dtype=np.int64),
        }
        table = pa.table(
            {
                k: pa.array(v.astype(np.float32).tolist(), type=pa.list_(pa.float32()))
                if v.ndim == 2
                else pa.array(v)
                for k, v in arrays.items()
            }
        )
        path = stage / meta.get_data_file_path(index)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        video = stage / meta.get_video_file_path(index, "observation.images.ego_view")
        video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(episode / "ego_view.mp4", video)
        stats = compute_episode_stats(arrays, {k: v for k, v in meta.features.items() if v["dtype"] != "video"})
        meta.save_episode(index, n, [TASK], stats)
        records.append(
            {
                "episode": index,
                "source_episode": read(episode / "parameters.json")["seed"],
                "frames": n,
                "simulation_run": str(episode),
                "quality": read(episode / "quality.json"),
                "video_sha256": digest(video),
                "physics_sha256": digest(episode / "physics.npz"),
                "commands_sha256": digest(episode / "commands.npz"),
            }
        )
    write(
        stage / "meta/provenance.json",
        {
            "kind": "physics_simulation",
            "split": split,
            "physics_grasp_verified": True,
            "real_demonstrations_used": False,
            "object_pose_assignment_after_reset": False,
            "episodes": records,
            "body_controller": "original SONIC low_latency encoder + decoder",
            "hand_control_hz": 50,
            "expert": "procedural IK with feedback correction; not a trained GR00T policy",
            "capture_schema": "inspire.basket_physics.v1",
            "note": "Training columns only; no PICO/SMPL fields fabricated.",
        },
    )
    (stage / "INCOMPLETE").unlink()
    stage.rename(output)
    return {
        "episodes": len(episodes),
        "frames": meta.total_frames,
        "path": str(output),
        "kind": "physics_simulation",
    }


def prepare(args):
    from .rollout import SCENE

    source_checkpoint = Path(args.checkpoint).expanduser().resolve()
    base_metadata = base_model_metadata(source_checkpoint)
    bundle = Path(args.output).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    targets = {f"sim_{part}": dataset_root / "sim" / bundle.name / part for part in ("train", "validation")}
    if any(p.exists() for p in targets.values()):
        raise FileExistsError("dataset already exported; run train.py without --execute to check it")
    if bundle.exists() and not (args.resume and (bundle / "INCOMPLETE").exists()):
        raise FileExistsError("choose a new output name or --resume an incomplete batch")
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "INCOMPLETE").write_text("Recording/checks pending; do not train.\n")
    runs = bundle / "recordings"
    runs.mkdir(exist_ok=True)
    sim_python = os.path.abspath(os.path.expanduser(args.sim_python))
    env = os.environ | {"MUJOCO_GL": "egl", "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
    accepted = []
    attempts = []
    for attempt in range(args.episodes * 4):
        seed = args.seed + attempt
        episode = runs / f"run_{attempt:03d}"
        command = [
            sim_python,
            "-B",
            "-m",
            "gear_sonic.devices.inspire.training.rollout",
            "episode",
            str(episode),
            "--seed",
            str(seed),
        ]
        if not episode.exists():
            with (runs / f"run_{attempt:03d}.log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"local rollout process failed; inspect {runs / f'run_{attempt:03d}.log'}")
        elif not (episode / "quality.json").exists():
            attempts.append(
                {"seed": seed, "path": str(episode), "passed": False, "reason": "interrupted attempt retained"}
            )
            continue
        quality = read(episode / "quality.json")
        attempts.append({"seed": seed, "path": str(episode), **quality})
        if quality["passed"]:
            if not (episode / "camera.json").exists():
                with (episode / "render.log").open("w") as log:
                    subprocess.run(
                        [
                            sim_python,
                            "-B",
                            "-m",
                            "gear_sonic.devices.inspire.training.rollout",
                            "render",
                            str(episode),
                        ],
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            accepted.append(episode)
        write(
            bundle / "progress.json", {"accepted": len(accepted), "requested": args.episodes, "attempts": attempts}
        )
        print(
            f"BASKET_RECORDING accepted={len(accepted)}/{args.episodes} attempt={attempt + 1} passed={quality['passed']}",
            flush=True,
        )
        if len(accepted) == args.episodes:
            break
    if len(accepted) != args.episodes:
        raise RuntimeError("insufficient successful physical grasps; batch remains INCOMPLETE")
    validation = max(1, args.episodes // 5)
    split = {"train": accepted[:-validation], "validation": accepted[-validation:]}
    robot = deployment("inspire", "sim_shared").robot_model()
    counts = {
        f"sim_{part}": export_split(targets[f"sim_{part}"], paths, robot, part) for part, paths in split.items()
    }
    # Retain the full scene and object/robot states alongside training data.
    # MuJoCo stays in the simulation environment; the exporter needs no driver.
    for target in targets.values():
        subprocess.run([args.sim_python, '-m', 'gear_sonic.devices.inspire.recording.simulation',
                        str(target), '--scene', str(SCENE)], cwd=ROOT, check=True)
    files = {
        str(p.relative_to(dataset_root)): digest(p)
        for root in targets.values()
        for p in root.rglob("*")
        if p.is_file()
    }
    manifest = {
        "schema": "inspire.basket_finetune.v2",
        "name": bundle.name,
        **base_metadata,
        "dataset_root": str(dataset_root),
        "datasets": {k: str(p.relative_to(dataset_root)) for k, p in targets.items()},
        "dataset_files_sha256": files,
        "counts": counts,
        "source_split": {
            part: [read(p / "parameters.json")["seed"] for p in paths] for part, paths in split.items()
        },
        "scene": str(SCENE),
        "scene_sha256": digest(SCENE),
        "seed": args.seed,
        "source_task": TASK,
        "requested_episodes": args.episodes,
        "physics_grasp_verified": True,
        "release_status": "DATA_BUILT_PREFLIGHT_PENDING",
        "real_demonstrations_used": False,
        "validation_scope": "Independent simulator resets; free body and free basket; final 2s bilateral unsupported lift >10cm. Simulation only.",
        "action_labels": "Actual SONIC token64 + adapter-submitted native6 targets; observations from the dynamic simulator.",
        "recordings": [str(p) for p in accepted],
    }
    write(bundle / "manifest.json", manifest)
    write_training_files(bundle, source_checkpoint, args.seed)
    write(
        bundle / "generation.json",
        {"status": "DATA_BUILT_PREFLIGHT_PENDING", "counts": counts, "optimizer_steps": 0},
    )
    print("BASKET_BUNDLE " + str(bundle), flush=True)
