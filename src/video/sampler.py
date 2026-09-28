"""Sequential clip sampling without loading an entire video into RAM."""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import cv2
from .reader import VideoReader

@dataclass
class SampledClip:
    frames: list
    frame_indices: list[int]
    start_time: float
    end_time: float


def iter_sampled_clips(video_path: str | Path, sample_rate: int = 10, clip_size: int = 16) -> Iterator[SampledClip]:
    if sample_rate < 1 or clip_size < 1:
        raise ValueError("sample_rate and clip_size must be positive")
    with VideoReader(video_path) as reader:
        fps = reader.fps
        indices: list[int] = []
        frames: list = []
        for idx, frame in reader.iter_frames(step=sample_rate):
            indices.append(idx)
            frames.append(frame)
            if len(frames) == clip_size:
                start = indices[0] / fps if fps else 0.0
                end = indices[-1] / fps if fps else start
                yield SampledClip(frames, indices, start, end)
                frames, indices = [], []
