import numpy as np
from src.preprocessing.preprocessing import preprocess_frame, preprocess_clip

def test_preprocessing_shapes():
    frame = np.zeros((100,120,3), dtype=np.uint8)
    assert tuple(preprocess_frame(frame).shape) == (3,224,224)
    assert tuple(preprocess_clip([frame]*16).shape) == (16,3,224,224)
