import cv2, numpy as np
from src.video.sampler import iter_sampled_clips

def test_sampler(tmp_path):
    p = tmp_path / "x.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (16,16))
    for _ in range(160): w.write(np.zeros((16,16,3), dtype=np.uint8))
    w.release()
    clips = list(iter_sampled_clips(p, sample_rate=1, clip_size=16))
    assert clips and len(clips[0].frames) == 16
