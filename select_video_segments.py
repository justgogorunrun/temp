import argparse
import base64
import json
from dataclasses import dataclass
from typing import List, Tuple

import cv2
from openai import OpenAI


@dataclass
class SegmentScore:
    segment_index: int
    frame_indices: List[int]
    score: float


def sample_frame_indices(total_frames: int, num_samples: int) -> List[int]:
    if total_frames <= 0:
        raise ValueError("Video has no frames.")
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if total_frames >= num_samples:
        step = (total_frames - 1) / (num_samples - 1)
        return [int(round(step * i)) for i in range(num_samples)]
    return [int(round(i * (total_frames - 1) / (num_samples - 1))) for i in range(num_samples)]


def read_frames(video_path: str, frame_indices: List[int]) -> List[Tuple[int, bytes]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    frames: List[Tuple[int, bytes]] = []
    current_index = 0
    target_set = set(frame_indices)
    max_index = max(frame_indices)

    while current_index <= max_index:
        ret, frame = cap.read()
        if not ret:
            break
        if current_index in target_set:
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                raise ValueError(f"Failed to encode frame {current_index}.")
            frames.append((current_index, buffer.tobytes()))
        current_index += 1

    cap.release()

    if len(frames) != len(frame_indices):
        raise ValueError("Could not read all requested frames.")

    frame_map = {idx: data for idx, data in frames}
    return [(idx, frame_map[idx]) for idx in frame_indices]


def chunk_frames(frames: List[Tuple[int, bytes]], num_segments: int) -> List[List[Tuple[int, bytes]]]:
    if len(frames) % num_segments != 0:
        raise ValueError("Number of frames must be divisible by number of segments.")
    segment_size = len(frames) // num_segments
    return [frames[i * segment_size : (i + 1) * segment_size] for i in range(num_segments)]


def encode_image_bytes(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def score_segment(client: OpenAI, question: str, segment_frames: List[Tuple[int, bytes]]) -> float:
    content = [
        {
            "type": "text",
            "text": (
                "You are scoring how relevant a video segment is to the question. "
                "Return JSON like {\"score\": 0.5} with score in [0,1]."
            ),
        }
    ]

    for _, frame_bytes in segment_frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_bytes(frame_bytes)},
            }
        )

    content.append(
        {
            "type": "text",
            "text": f"Question: {question}\nScore this segment.",
        }
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=100,
        temperature=0,
    )

    message = response.choices[0].message.content or ""
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        data = json.loads(message.strip().split("\n")[-1])

    score = float(data.get("score", 0))
    return max(0.0, min(1.0, score))


def select_top_k_segments(video_path: str, question: str, k: int) -> List[SegmentScore]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_indices = sample_frame_indices(total_frames, 256)
    frames = read_frames(video_path, frame_indices)
    segments = chunk_frames(frames, 16)

    client = OpenAI()
    scores: List[SegmentScore] = []

    for segment_index, segment_frames in enumerate(segments):
        score = score_segment(client, question, segment_frames)
        scores.append(
            SegmentScore(
                segment_index=segment_index,
                frame_indices=[idx for idx, _ in segment_frames],
                score=score,
            )
        )

    scores.sort(key=lambda item: item.score, reverse=True)
    return scores[:k]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select top-k video segments relevant to a question using GPT-4o."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument("--question", required=True, help="Question text")
    parser.add_argument("--k", type=int, required=True, help="Number of top segments to return")
    args = parser.parse_args()

    top_segments = select_top_k_segments(args.video, args.question, args.k)
    for item in top_segments:
        print(
            json.dumps(
                {
                    "segment_index": item.segment_index,
                    "score": item.score,
                    "frame_indices": item.frame_indices,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
