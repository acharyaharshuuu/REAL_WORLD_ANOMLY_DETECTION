"""Small OpenCV video reader wrapper."""
from pathlib import Path
from typing import Iterator, Optional, Tuple
import cv2

class VideoReader:
    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        self.cap: Optional[cv2.VideoCapture] = None
        self._open()

    def _open(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

    @property
    def fps(self) -> float:
        return float(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap else 0.0

    @property
    def frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap else 0

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap else 0

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap else 0

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def read_frame(self, frame_index: Optional[int] = None):
        if self.cap is None:
            raise RuntimeError("Video reader is closed")
        if frame_index is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def iter_frames(self, start: int = 0, step: int = 1) -> Iterator[Tuple[int, object]]:
        if self.cap is None:
            raise RuntimeError("Video reader is closed")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        index = start
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
            if step > 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, index + step - 1)
                index += step - 1

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
