from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


@dataclass
class ShotBoundary:
    start_frame: int
    end_frame: int


class DINOv3ShotSegmenter:
    """Use DINOv3 frame embeddings to segment a video into shots."""

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        threshold: float = 0.80,
        min_shot_len: int = 1,
        batch_size: int = 16,
        device: str = "cuda:0",
        visualize: bool = False,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.min_shot_len = min_shot_len
        self.batch_size = batch_size
        self.visualize = visualize

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def sample_frames_1fps(video_path: str | Path) -> Tuple[List[int], List[Image.Image], float]:
        """Read frames at 1 FPS and return (frame_ids, PIL_frames, src_fps)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        step = max(int(round(fps)), 1)

        frame_ids: List[int] = []
        frames: List[Image.Image] = []
        frame_idx = 0

        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
                frame_ids.append(frame_idx)
            frame_idx += 1

        cap.release()
        return frame_ids, frames, float(fps)

    def extract_embeddings(self, frames: Sequence[Image.Image]) -> torch.Tensor:
        if not frames:
            return torch.empty(0, 1, device=self.device)

        all_embeddings = []
        with torch.inference_mode():
            for i in range(0, len(frames), self.batch_size):
                batch_frames = frames[i : i + self.batch_size]
                inputs = self.processor(images=list(batch_frames), return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    emb = outputs.pooler_output
                else:
                    emb = outputs.last_hidden_state[:, 0, :]
                emb = F.normalize(emb, dim=-1)
                all_embeddings.append(emb)

        return torch.cat(all_embeddings, dim=0)

    def segment_video(self, video_path: str | Path) -> List[ShotBoundary]:
        frame_ids, frames, _ = self.sample_frames_1fps(video_path)
        if not frame_ids:
            return []

        if len(frame_ids) == 1:
            return [ShotBoundary(start_frame=frame_ids[0], end_frame=frame_ids[0])]

        embeddings = self.extract_embeddings(frames)
        sims = F.cosine_similarity(embeddings[:-1], embeddings[1:], dim=-1).detach().cpu().numpy()

        boundaries = [0]
        for i, sim in enumerate(sims, start=1):
            if sim < self.threshold:
                boundaries.append(i)
        boundaries.append(len(frame_ids))

        shots: List[ShotBoundary] = []
        for start_i, end_i in zip(boundaries[:-1], boundaries[1:]):
            if end_i - start_i < self.min_shot_len:
                continue
            shots.append(
                ShotBoundary(
                    start_frame=frame_ids[start_i],
                    end_frame=frame_ids[end_i - 1],
                )
            )

        if self.visualize:
            self._plot_similarities(video_path, sims)

        return shots

    def _plot_similarities(self, video_path: str | Path, sims: np.ndarray) -> None:
        plt.figure(figsize=(12, 4))
        plt.plot(np.arange(len(sims)), sims, label="adjacent cosine similarity")
        plt.axhline(self.threshold, color="red", linestyle="--", label=f"threshold={self.threshold}")
        plt.title(f"Shot similarity curve: {Path(video_path).name}")
        plt.xlabel("sample index @1fps")
        plt.ylabel("cosine similarity")
        plt.legend()
        plt.tight_layout()
        plt.show()


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment one video into shots with DINOv3.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--threshold", type=float, default=0.80, help="Boundary threshold on adjacent cosine similarity")
    parser.add_argument("--min-shot-len", type=int, default=1, help="Min shot length measured on 1fps sampled frames")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-name", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--visualize", action="store_true", help="Enable shot score visualization")
    return parser.parse_args()


def main() -> None:
    args = _build_args()
    segmenter = DINOv3ShotSegmenter(
        model_name=args.model_name,
        threshold=args.threshold,
        min_shot_len=args.min_shot_len,
        batch_size=args.batch_size,
        device=args.device,
        visualize=args.visualize,
    )
    shots = segmenter.segment_video(args.video)
    print([shot.__dict__ for shot in shots])


if __name__ == "__main__":
    main()
