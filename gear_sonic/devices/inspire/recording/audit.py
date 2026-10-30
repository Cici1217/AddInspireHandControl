"""Check standalone recordings and batches without weakening per-episode checks."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from ..config import DEFAULT_MAPPING_PATH, load_mapping

from .schema import (
    CONTROL_FPS,
    LEROBOT_TIMING_PATH,
    LEROBOT_TIMING_SCHEMA,
    TIMING_FIELDS,
    lerobot_features,
    lerobot_modality_config,
    validate_capture_identity,
)


@dataclass(frozen=True)
class LeRobotAuditReport:
    dataset: str
    episodes: int
    frames: int
    video_frames: int
    fps: float
    task_prompt: str
    stats_present: bool
    validation_level: str
    timing_trace_present: bool
    source_span_seconds: float
    source_camera_unique_frames: int
    source_hand_sequence_repeats: int
    source_teleop_target_unique_frames: int

    def as_dict(self):
        return asdict(self)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _one_jsonl(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(f"expected one metadata entry: {path}")
    return rows[0]


def _timing(root, capture, frames, episode_index=0):
    path = (f"meta/recordings/episode_{episode_index:06d}/timing.npz"
            if (root / "meta/capture_batch.json").exists() else LEROBOT_TIMING_PATH)
    expected = {"schema": LEROBOT_TIMING_SCHEMA, "path": path, "frames": frames}
    if capture.get("timing_trace") != expected:
        raise ValueError("capture timing_trace descriptor is incompatible")
    arrays = {}
    with np.load(root / path, allow_pickle=False) as trace:
        if str(trace.get("schema")) != LEROBOT_TIMING_SCHEMA:
            raise ValueError("capture timing trace has an unknown schema")
        for key in TIMING_FIELDS:
            value = np.asarray(trace[key])
            if value.shape != (frames,) or value.dtype.kind not in "iu" or np.any(value < 0):
                raise ValueError(f"invalid timing trace: {key}")
            arrays[key] = value.astype(np.int64)
            difference = np.diff(arrays[key])
            if np.any(difference <= 0 if key == "sample_monotonic_ns" else difference < 0):
                raise ValueError(f"capture {key} regressed or repeated sample time")
    camera_unique = int(np.count_nonzero(np.r_[True, np.diff(arrays["camera_source_timestamp_ns"]) != 0]))
    hand_repeats = int(np.count_nonzero(np.diff(arrays["hand_command_sequence"]) == 0))
    target_unique = int(np.count_nonzero(np.r_[True, np.diff(arrays["teleop_target_frame_index"]) != 0]))
    summary = capture["summary"]
    if hand_repeats != summary["hand_sequence_repeats"]:
        raise ValueError("timing and summary hand repeat counts disagree")
    if camera_unique > summary["camera_unique_frames"] or target_unique > summary["teleop_target_unique_frames"]:
        raise ValueError("timing has more unique frames than the source summary")
    span = float(arrays["sample_monotonic_ns"][-1] - arrays["sample_monotonic_ns"][0]) / 1e9
    return span, camera_unique, hand_repeats, target_unique


def _audit_episode(dataset, *, episode_index=0, robot_model=None, config_path=DEFAULT_MAPPING_PATH):
    root = Path(dataset).expanduser().resolve()
    if (root / "INCOMPLETE").exists():
        raise ValueError("dataset is marked INCOMPLETE")
    if robot_model is None:
        from ..model import instantiate_g1_rh56dfx_robot_model

        robot_model = instantiate_g1_rh56dfx_robot_model(config_path=config_path)
    try:
        import av
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("dataset check needs the data_collection environment") from exc

    info = _json(root / "meta/info.json")
    from gear_sonic.utils.data_collection.batch import episode_metadata
    capture = _json(episode_metadata(root, episode_index, "inspire_capture.json"))
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    if len(episodes) != info["total_episodes"] or [e["episode_index"] for e in episodes] != list(range(len(episodes))):
        raise ValueError("episode metadata indices disagree")
    episode = episodes[episode_index]
    task = _one_jsonl(root / "meta/tasks.jsonl")
    features = lerobot_features(robot_model)
    for key, feature in features.items():
        actual = info["features"].get(key, {})
        if actual.get("dtype") != feature["dtype"] or actual.get("shape") != list(feature["shape"]):
            raise ValueError(f"incompatible feature: {key}")
    if _json(root / "meta/modality.json") != lerobot_modality_config(robot_model):
        raise ValueError("incompatible Inspire modality")
    validate_capture_identity(capture, config_path=config_path)
    frames = episode["length"]
    if (
        sum(e["length"] for e in episodes) != info["total_frames"]
        or frames < 40
        or episode["length"] != frames
        or capture["actual_frames"] != frames
    ):
        raise ValueError("expected a complete episode of at least 40 frames")
    if info["fps"] != CONTROL_FPS or capture["fps"] != CONTROL_FPS:
        raise ValueError("dataset fps must be 50")
    if not str(task.get("task", "")).strip():
        raise ValueError("dataset task prompt is empty")
    summary = capture["summary"]
    requested = summary["requested_frames"]
    duration_limited = summary.get("stop_reason") == "duration_limit"
    if summary["written_frames"] != frames or (
        requested < frames if duration_limited else requested != frames
    ):
        raise ValueError("capture summary frame counts disagree")
    if summary["duration_seconds"] <= 0 or any(count < 0 for count in summary["drops"].values()):
        raise ValueError("invalid capture duration or drop counts")
    timing = _timing(root, capture, frames, episode_index)

    parquet = list(root.glob("data/**/*.parquet"))
    video = list(root.glob("videos/**/episode_*.mp4"))
    if len(parquet) != info["total_episodes"] or len(video) != info["total_episodes"]:
        raise ValueError("expected one parquet and one ego_view video per episode")
    fields = dict(episode_chunk=episode_index // info.get("chunks_size", 1000), episode_index=episode_index, video_key="observation.images.ego_view")
    data_path = root / info["data_path"].format(**fields)
    video_path = root / info["video_path"].format(**fields)
    if not data_path.resolve().is_relative_to(root) or not video_path.resolve().is_relative_to(root):
        raise ValueError("episode path escapes dataset")
    table = pq.read_table(data_path)
    if not np.all(np.asarray(table["episode_index"].to_pylist()) == episode_index):
        raise ValueError("parquet episode index disagrees")
    if table.num_rows != frames:
        raise ValueError("parquet frame count disagrees")
    columns = {}
    for key, feature in features.items():
        if feature["dtype"] == "video":
            continue
        value = np.asarray(table[key].to_pylist(), dtype=np.float32)
        shape = tuple(feature["shape"])
        if shape == (1,) and value.shape == (frames,):
            value = value.reshape(frames, 1)
        if value.shape != (frames, *shape) or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid parquet column: {key}")
        columns[key] = value
    if not np.array_equal(table["frame_index"].to_pylist(), np.arange(frames)):
        raise ValueError("frame_index must be contiguous from zero")
    if not np.allclose(table["timestamp"].to_pylist(), np.arange(frames) / CONTROL_FPS, rtol=1e-3, atol=1e-6):
        raise ValueError("timestamps do not match nominal 50fps")
    mapping = load_mapping(config_path)
    for side in ("left", "right"):
        for row in columns[f"action.{side}_hand_joints"]:
            mapping.validate_limits(row, field_name=f"action.{side}_hand_joints", side=side)
    for key in ("observation.hand_gateway_healthy", "teleop.hand_target_valid", "teleop.hand_command_sent"):
        if np.any((columns[key] != 0) & (columns[key] != 1)):
            raise ValueError(f"invalid flags: {key}")
    with av.open(str(video_path)) as container:
        video_frames = sum(1 for _ in container.decode(video=0))
    if video_frames != frames:
        raise ValueError(f"video has {video_frames} frames, expected {frames}")
    return LeRobotAuditReport(
        str(root),
        1,
        frames,
        video_frames,
        CONTROL_FPS,
        task["task"],
        (root / "meta/stats.json").is_file(),
        capture["validation_level"],
        True,
        *timing,
    )


def audit_lerobot_dataset(dataset, *, robot_model=None, config_path=DEFAULT_MAPPING_PATH):
    root = Path(dataset).expanduser().resolve()
    info = _json(root / 'meta/info.json')
    count = info['total_episodes']
    if type(count) is not int or count < 1:
        raise ValueError('dataset has no complete episodes')
    if count == 1 and not (root / 'meta/capture_batch.json').exists():
        return _audit_episode(root, robot_model=robot_model, config_path=config_path)
    from gear_sonic.utils.data_collection.batch import SCHEMA
    batch = _json(root / 'meta/capture_batch.json')
    if batch.get('schema') != SCHEMA or len(batch['episodes']) != count:
        raise ValueError('capture batch metadata disagrees')
    if robot_model is None:
        from ..model import instantiate_g1_rh56dfx_robot_model
        robot_model = instantiate_g1_rh56dfx_robot_model(config_path=config_path)
    reports = [_audit_episode(root, episode_index=i, robot_model=robot_model, config_path=config_path)
               for i in range(count)]
    for index, (entry, report) in enumerate(zip(batch['episodes'], reports)):
        metadata = _json(root / f'meta/recordings/episode_{index:06d}/end_effector.json')
        if (entry['episode_index'] != index or entry['frames'] != report.frames
                or entry['outcome'] != metadata['outcome']):
            raise ValueError('batch episode outcome or frame count disagrees')
    fields = reports[0].as_dict()
    fields['episodes'] = count
    for key in ('frames', 'video_frames', 'source_span_seconds', 'source_camera_unique_frames',
                'source_hand_sequence_repeats', 'source_teleop_target_unique_frames'):
        fields[key] = sum(getattr(report, key) for report in reports)
    if _json(root / 'meta/inspire_capture.json')['actual_frames'] != fields['frames']:
        raise ValueError('capture batch total frames disagree')
    return LeRobotAuditReport(**fields)
