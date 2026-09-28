"""Preprocessing compatible with pretrained ImageNet ResNet18."""
import cv2
import torch
from PIL import Image
from torchvision import transforms
from src.utils.config import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD

_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
])

def preprocess_frame(frame_bgr) -> torch.Tensor:
    if frame_bgr is None:
        raise ValueError("Frame is None")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return _TRANSFORM(Image.fromarray(frame_rgb))

def preprocess_clip(frames) -> torch.Tensor:
    if not frames:
        raise ValueError("Clip is empty")
    return torch.stack([preprocess_frame(frame) for frame in frames])
