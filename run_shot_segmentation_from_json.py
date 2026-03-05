from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import torch
from tqdm import tqdm

from shot_segmentation import DINOv3ShotSegmenter


def load_json(input_json: str | Path) -> Any:
    with open(input_json, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_records(data: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Normalize user json into list[dict] and preserve source schema info."""
    schema = {"kind": "list_dict", "wrapped_key": None}

    if isinstance(data, list):
        if not data:
            return [], schema
        if isinstance(data[0], str):
            schema["kind"] = "list_str"
            return [{"video_id": x} for x in data], schema
        if isinstance(data[0], dict):
            return data, schema
        raise ValueError("Unsupported list item type in input JSON.")

    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            schema = {"kind": "dict_wrapped", "wrapped_key": "data", "base": data}
            return data["data"], schema
        raise ValueError("Unsupported dict schema. Expected {'data': [...]}.")

    raise ValueError("Unsupported json format. Expected list[str], list[dict], or {'data': [...]}.")


def dump_records(records: List[Dict[str, Any]], output_json: str | Path, schema: Dict[str, Any], dataset_stats: Dict[str, Any]) -> None:
    if schema["kind"] == "dict_wrapped":
        base = dict(schema["base"])
        base[schema["wrapped_key"]] = records
        base["dataset_shot_statistics"] = dataset_stats
        out_obj: Any = base
    else:
        out_obj = records

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)


def dump_dataset_stats(output_json: str | Path, dataset_stats: Dict[str, Any]) -> Path:
    out_path = Path(output_json)
    stats_path = out_path.with_name(f"{out_path.stem}_dataset_stats{out_path.suffix}")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, ensure_ascii=False, indent=2)
    return stats_path


def get_video_path(video_root: Path, video_id: str, suffix: str) -> Path:
    return video_root / f"{video_id}{suffix}"


def probe_video_metadata(video_path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if fps > 0 else 0.0
    cap.release()
    return {
        "video_fps": round(fps, 4),
        "video_total_frames": total_frames,
        "video_duration_sec": round(duration_sec, 4),
    }


def build_shot_stats(shots: List[Dict[str, int]], video_fps: float) -> Dict[str, Any]:
    lengths = [(s["end_frame"] - s["start_frame"] + 1) for s in shots if s["end_frame"] >= s["start_frame"]]
    duration_lengths = [l / max(video_fps, 1e-6) for l in lengths]

    if not lengths:
        return {
            "shot_count": 0,
            "shot_max_frames": 0,
            "shot_min_frames": 0,
            "shot_avg_frames": 0.0,
            "shot_median_frames": 0.0,
            "shot_std_frames": 0.0,
            "shot_max_duration_sec": 0.0,
            "shot_min_duration_sec": 0.0,
            "shot_avg_duration_sec": 0.0,
            "shot_median_duration_sec": 0.0,
        }

    avg_frames = mean(lengths)
    std_frames = (sum((x - avg_frames) ** 2 for x in lengths) / len(lengths)) ** 0.5
    return {
        "shot_count": len(lengths),
        "shot_max_frames": max(lengths),
        "shot_min_frames": min(lengths),
        "shot_avg_frames": round(avg_frames, 4),
        "shot_median_frames": round(median(lengths), 4),
        "shot_std_frames": round(std_frames, 4),
        "shot_max_duration_sec": round(max(duration_lengths), 4),
        "shot_min_duration_sec": round(min(duration_lengths), 4),
        "shot_avg_duration_sec": round(mean(duration_lengths), 4),
        "shot_median_duration_sec": round(median(duration_lengths), 4),
    }


def process_one_record(record: Dict[str, Any], segmenter: DINOv3ShotSegmenter, video_root: Path, video_id_key: str, video_suffix: str) -> Dict[str, Any]:
    r = dict(record)
    video_id = str(r[video_id_key])
    video_path = get_video_path(video_root, video_id, video_suffix)

    if not video_path.exists():
        r["shots"] = []
        r["shot_error"] = f"video not found: {video_path}"
        return r

    try:
        meta = probe_video_metadata(video_path)
        shots_obj = segmenter.segment_video(video_path)
        shots = [{"start_frame": s.start_frame, "end_frame": s.end_frame} for s in shots_obj]

        r["shots"] = shots
        r.update(meta)
        r.update(build_shot_stats(shots, float(meta["video_fps"])))
        r["shot_density_per_min"] = round((r["shot_count"] / max(meta["video_duration_sec"] / 60.0, 1e-6)), 4)
        r["shot_coverage_ratio"] = round(
            (sum((s["end_frame"] - s["start_frame"] + 1) for s in shots) / max(meta["video_total_frames"], 1)),
            4,
        )
    except Exception as e:  # noqa: BLE001
        r["shots"] = []
        r["shot_error"] = str(e)

    return r


def process_subset(
    rank: int,
    gpu_id: int,
    subset: Sequence[Dict[str, Any]],
    video_root: str,
    video_id_key: str,
    video_suffix: str,
    model_name: str,
    threshold: float,
    min_shot_len: int,
    batch_size: int,
    visualize: bool,
    queue: mp.Queue,
) -> None:
    segmenter = DINOv3ShotSegmenter(
        model_name=model_name,
        threshold=threshold,
        min_shot_len=min_shot_len,
        batch_size=batch_size,
        device=f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu",
        visualize=visualize,
    )

    outputs: List[Tuple[int, Dict[str, Any]]] = []
    root = Path(video_root)

    for i, item in enumerate(tqdm(subset, desc=f"GPU-{gpu_id}", position=rank, leave=True)):
        outputs.append((i, process_one_record(item, segmenter, root, video_id_key, video_suffix)))

    queue.put((rank, outputs))


def split_even(items: Sequence[Dict[str, Any]], parts: int) -> List[List[Dict[str, Any]]]:
    chunks = [[] for _ in range(parts)]
    for i, item in enumerate(items):
        chunks[i % parts].append(item)
    return chunks


def _safe_mean(values: List[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def summarize_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in records if not r.get("shot_error")]
    invalid = [r for r in records if r.get("shot_error")]

    shot_counts = [int(r.get("shot_count", 0)) for r in valid]
    durations = [float(r.get("video_duration_sec", 0.0)) for r in valid]
    shot_avg_frames = [float(r.get("shot_avg_frames", 0.0)) for r in valid]
    shot_max_frames = [int(r.get("shot_max_frames", 0)) for r in valid]
    shot_min_frames = [int(r.get("shot_min_frames", 0)) for r in valid if int(r.get("shot_count", 0)) > 0]
    density = [float(r.get("shot_density_per_min", 0.0)) for r in valid]

    total_shots = sum(shot_counts)
    avg_shot_frames_weighted = 0.0
    if total_shots > 0:
        total_shot_frames = sum(float(r.get("shot_avg_frames", 0.0)) * int(r.get("shot_count", 0)) for r in valid)
        avg_shot_frames_weighted = round(total_shot_frames / total_shots, 4)

    return {
        "num_videos_total": len(records),
        "num_videos_success": len(valid),
        "num_videos_failed": len(invalid),
        "video_duration_sec": {
            "avg": _safe_mean(durations),
            "min": round(min(durations), 4) if durations else 0.0,
            "max": round(max(durations), 4) if durations else 0.0,
            "median": round(median(durations), 4) if durations else 0.0,
        },
        "shot_count": {
            "avg": _safe_mean([float(x) for x in shot_counts]),
            "min": min(shot_counts) if shot_counts else 0,
            "max": max(shot_counts) if shot_counts else 0,
            "median": round(median(shot_counts), 4) if shot_counts else 0.0,
            "total": total_shots,
        },
        "shot_max_frames": {
            "avg": _safe_mean([float(x) for x in shot_max_frames]),
            "min": min(shot_max_frames) if shot_max_frames else 0,
            "max": max(shot_max_frames) if shot_max_frames else 0,
        },
        "shot_min_frames": {
            "avg": _safe_mean([float(x) for x in shot_min_frames]),
            "min": min(shot_min_frames) if shot_min_frames else 0,
            "max": max(shot_min_frames) if shot_min_frames else 0,
        },
        "shot_avg_frames": {
            "avg": _safe_mean(shot_avg_frames),
            "min": round(min(shot_avg_frames), 4) if shot_avg_frames else 0.0,
            "max": round(max(shot_avg_frames), 4) if shot_avg_frames else 0.0,
            "weighted_avg_by_shots": avg_shot_frames_weighted,
        },
        "shot_density_per_min": {
            "avg": _safe_mean(density),
            "min": round(min(density), 4) if density else 0.0,
            "max": round(max(density), 4) if density else 0.0,
            "median": round(median(density), 4) if density else 0.0,
        },
    }


def print_dataset_summary(stats: Dict[str, Any]) -> None:
    print("\n===== Dataset Shot Statistics =====")
    print(f"Videos: total={stats['num_videos_total']} success={stats['num_videos_success']} failed={stats['num_videos_failed']}")
    print(f"Video duration(sec): {stats['video_duration_sec']}")
    print(f"Shot count: {stats['shot_count']}")
    print(f"Shot max frames: {stats['shot_max_frames']}")
    print(f"Shot min frames: {stats['shot_min_frames']}")
    print(f"Shot avg frames: {stats['shot_avg_frames']}")
    print(f"Shot density(/min): {stats['shot_density_per_min']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch shot segmentation by reading video IDs from a json file.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--video-root", default="/mnt/bn/laion400m/zhangkc/longva_train_data/shot2story/videos/release_134k_videos", help="Root folder of videos")
    parser.add_argument("--video-id-key", default="video", help="json key name of the video id")
    parser.add_argument("--video-suffix", default=".mp4", help="video file suffix, e.g. .mp4")
    parser.add_argument("--model-name", default="/mnt/bn/laion400m/zhangkc/dinov3-vit7b16-pretrain-lvd1689m")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--min-shot-len", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-gpus", type=int, default=8, help="How many GPUs to use; default is all visible GPUs")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.input_json)
    records, schema = normalize_records(data)

    if not records:
        empty_stats = summarize_dataset([])
        dump_records([], args.output_json, schema, empty_stats)
        stats_path = dump_dataset_stats(args.output_json, empty_stats)
        print(f"No records found in input json. Saved empty stats to {stats_path}")
        return

    if args.video_id_key not in records[0]:
        raise KeyError(f"video_id key '{args.video_id_key}' not found in input records")

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("CUDA is unavailable, fallback to single-process CPU mode.")
        segmenter = DINOv3ShotSegmenter(
            model_name=args.model_name,
            threshold=args.threshold,
            min_shot_len=args.min_shot_len,
            batch_size=args.batch_size,
            device="cpu",
            visualize=args.visualize,
        )
        out_records = []
        for record in tqdm(records, desc="CPU"):
            out_records.append(
                process_one_record(
                    record,
                    segmenter,
                    Path(args.video_root),
                    args.video_id_key,
                    args.video_suffix,
                )
            )

        dataset_stats = summarize_dataset(out_records)
        dump_records(out_records, args.output_json, schema, dataset_stats)
        stats_path = dump_dataset_stats(args.output_json, dataset_stats)
        print_dataset_summary(dataset_stats)
        print(f"Saved updated records to: {args.output_json}")
        print(f"Saved dataset stats to: {stats_path}")
        return

    num_workers = args.num_gpus if args.num_gpus is not None else available_gpus
    num_workers = max(1, min(num_workers, available_gpus, len(records)))
    gpu_ids = list(range(num_workers))

    chunks = split_even(records, num_workers)
    queue: mp.Queue = mp.Queue()

    procs: List[mp.Process] = []
    for rank, (gpu_id, subset) in enumerate(zip(gpu_ids, chunks)):
        p = mp.Process(
            target=process_subset,
            args=(
                rank,
                gpu_id,
                subset,
                args.video_root,
                args.video_id_key,
                args.video_suffix,
                args.model_name,
                args.threshold,
                args.min_shot_len,
                args.batch_size,
                args.visualize,
                queue,
            ),
        )
        p.start()
        procs.append(p)

    gathered: List[List[Dict[str, Any]]] = [None] * num_workers  # type: ignore[list-item]
    for _ in range(num_workers):
        rank, outputs = queue.get()
        outputs = [record for _, record in sorted(outputs, key=lambda x: x[0])]
        gathered[rank] = outputs

    for p in procs:
        p.join()

    out_records: List[Dict[str, Any]] = []
    max_len = max(len(x) for x in gathered)
    for i in range(max_len):
        for worker_records in gathered:
            if i < len(worker_records):
                out_records.append(worker_records[i])

    dataset_stats = summarize_dataset(out_records)
    dump_records(out_records, args.output_json, schema, dataset_stats)
    stats_path = dump_dataset_stats(args.output_json, dataset_stats)

    print_dataset_summary(dataset_stats)
    print(f"Saved updated records to: {args.output_json}")
    print(f"Saved dataset stats to: {stats_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
