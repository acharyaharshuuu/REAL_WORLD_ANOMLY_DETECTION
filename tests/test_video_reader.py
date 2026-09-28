from pathlib import Path
import cv2, numpy as np
from src.video.reader import VideoReader

def make_video(path: Path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    for _ in range(20): writer.write(np.zeros((24,32,3), dtype=np.uint8))
    writer.release()

def test_video_reader(tmp_path):
    p = tmp_path / "x.mp4"; make_video(p)
    with VideoReader(p) as r:
        assert r.frame_count == 20
        assert r.width == 32 and r.height == 24
        assert r.fps > 0
        assert r.read_frame(0) is not None
