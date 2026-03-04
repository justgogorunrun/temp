from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm

from shot_segmentation import DINOv3ShotSegmenter


def load_records(input_json: str | Path) -> Tuple[List[Dict[str, Any]], bool]:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if data and isinstance(data[0], str):
            return [{"video_id": x} for x in data], True
        if data and isinstance(data[0], dict):
            return data, False
        return [], False

    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"], False

    raise ValueError("Unsupported json format. Expected list[str], list[dict], or {'data': [...]}.")


def dump_records(records: List[Dict[str, Any]], input_json: str | Path, output_json: str | Path, wrapped: bool = False) -> None:
    out_obj: Any = records
    if wrapped:
        out_obj = {"data": records}
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)


def get_video_path(video_root: Path, video_id: str, suffix: str) -> Path:
    return video_root / f"{video_id}{suffix}"


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
        record = dict(item)
        video_id = str(record[video_id_key])
        video_path = get_video_path(root, video_id, video_suffix)

        if not video_path.exists():
            record["shots"] = []
            record["shot_error"] = f"video not found: {video_path}"
            outputs.append((i, record))
            continue

        try:
            shots = segmenter.segment_video(video_path)
            record["shots"] = [{"start_frame": s.start_frame, "end_frame": s.end_frame} for s in shots]
        except Exception as e:  # noqa: BLE001
            record["shots"] = []
            record["shot_error"] = str(e)

        outputs.append((i, record))

    queue.put((rank, outputs))


def split_even(items: Sequence[Dict[str, Any]], parts: int) -> List[List[Dict[str, Any]]]:
    chunks = [[] for _ in range(parts)]
    for i, item in enumerate(items):
        chunks[i % parts].append(item)
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch shot segmentation by reading video IDs from a json file.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--video-root", required=True, help="Root folder of videos")
    parser.add_argument("--video-id-key", default="video_id", help="json key name of the video id")
    parser.add_argument("--video-suffix", default=".mp4", help="video file suffix, e.g. .mp4")
    parser.add_argument("--model-name", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--min-shot-len", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-gpus", type=int, default=None, help="How many GPUs to use; default is all visible GPUs")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, _ = load_records(args.input_json)

    if not records:
        dump_records([], args.input_json, args.output_json)
        print("No records found in input json.")
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
            r = dict(record)
            video_path = get_video_path(Path(args.video_root), str(r[args.video_id_key]), args.video_suffix)
            try:
                shots = segmenter.segment_video(video_path)
                r["shots"] = [{"start_frame": s.start_frame, "end_frame": s.end_frame} for s in shots]
            except Exception as e:  # noqa: BLE001
                r["shots"] = []
                r["shot_error"] = str(e)
            out_records.append(r)
        dump_records(out_records, args.input_json, args.output_json)
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

    dump_records(out_records, args.input_json, args.output_json)
    print(f"Saved updated records to: {args.output_json}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
