"""Frame/clip feature extraction using ResNet18."""
import torch
from src.models.cnn import ResNet18FeatureExtractor
from src.preprocessing.preprocessing import preprocess_clip

class CNNFeatureExtractor:
    def __init__(self, device: str | None = None, pretrained: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ResNet18FeatureExtractor(pretrained=pretrained).to(self.device).eval()

    @torch.no_grad()
    def extract_clip(self, frames) -> torch.Tensor:
        images = preprocess_clip(frames).to(self.device)
        features = self.model(images)
        return features.cpu()

    @torch.no_grad()
    def extract_batch(self, clips: list[list]) -> torch.Tensor:
        outputs = [self.extract_clip(frames) for frames in clips]
        return torch.stack(outputs) if outputs else torch.empty((0, 0, 512))
