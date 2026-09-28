"""Production inference pipeline."""
import json
from pathlib import Path
import torch

from src.models.cnn_lstm import CNNLSTMClassifier
from src.features.cnn_features import CNNFeatureExtractor
from src.video.sampler import iter_sampled_clips
from src.utils import config

class InferenceEngine:
    def __init__(self, model_path=None, class_names_path=None, model_config_path=None, device=None):
        self.model_path = Path(model_path or config.MODEL_PATH)
        self.class_names_path = Path(class_names_path or config.CLASS_NAMES_PATH)
        self.model_config_path = Path(model_config_path or config.MODEL_CONFIG_PATH)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {self.model_path}")
        if not self.class_names_path.exists() or not self.model_config_path.exists():
            raise FileNotFoundError("class_names.json and model_config.json are required")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_names = json.loads(self.class_names_path.read_text(encoding="utf-8"))
        cfg = json.loads(self.model_config_path.read_text(encoding="utf-8"))
        self.cfg = cfg
        self.model = CNNLSTMClassifier(
            input_size=cfg["feature_size"], hidden_size=cfg["lstm_hidden_size"],
            num_layers=cfg["lstm_layers"], num_classes=len(self.class_names),
            dropout=cfg.get("dropout", 0.0)
        ).to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        self.model.eval()
        self.extractor = CNNFeatureExtractor(device=str(self.device), pretrained=True)

    @torch.no_grad()
    def predict_video(self, video_path):
        results = []
        for clip in iter_sampled_clips(video_path, self.cfg["sample_rate"], self.cfg["clip_size"]):
            features = self.extractor.extract_clip(clip.frames).unsqueeze(0).to(self.device)
            logits = self.model(features)
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs))
            results.append({
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "activity": self.class_names[idx],
                "confidence": float(probs[idx]),
            })
        return results
