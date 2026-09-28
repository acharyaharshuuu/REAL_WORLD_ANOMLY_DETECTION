"""Central project configuration.

Values are intentionally conservative defaults and should be aligned with the
existing notebooks once they are available in the repository.
"""
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = Path(os.getenv("DATASET_PATH", ROOT_DIR / "data" / "raw" / "UCF_Crime"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "models" / "trained" / "cnn_lstm_model.pth"))
CLASS_NAMES_PATH = Path(os.getenv("CLASS_NAMES_PATH", ROOT_DIR / "models" / "trained" / "class_names.json"))
MODEL_CONFIG_PATH = Path(os.getenv("MODEL_CONFIG_PATH", ROOT_DIR / "models" / "trained" / "model_config.json"))

CLIP_SIZE = int(os.getenv("CLIP_SIZE", "16"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "10"))
FEATURE_SIZE = int(os.getenv("FEATURE_SIZE", "512"))
LSTM_HIDDEN_SIZE = int(os.getenv("LSTM_HIDDEN_SIZE", "256"))
LSTM_LAYERS = int(os.getenv("LSTM_LAYERS", "1"))
DROPOUT = float(os.getenv("DROPOUT", "0.0"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
MIN_CONSECUTIVE_CLIPS = int(os.getenv("MIN_CONSECUTIVE_CLIPS", "2"))

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
