"""Train CNN+LSTM from the real dataset without extracting the whole dataset to JPGs.

The dataset layout is expected to be class directories containing videos, e.g.:
data/raw/UCF_Crime/<ClassName>/*.mp4

If the existing notebooks use a different layout or split policy, align the
configuration after supplying those notebooks; this script deliberately avoids
inventing notebook-specific values.
"""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.video.sampler import iter_sampled_clips
from src.features.cnn_features import CNNFeatureExtractor
from src.models.cnn_lstm import CNNLSTMClassifier
from src.utils import config

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

class VideoClipDataset(Dataset):
    def __init__(self, records, class_to_idx, sample_rate, clip_size, device):
        self.records = records
        self.class_to_idx = class_to_idx
        self.sample_rate = sample_rate
        self.clip_size = clip_size
        self.extractor = CNNFeatureExtractor(device=device, pretrained=True)

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        clips = iter_sampled_clips(path, self.sample_rate, self.clip_size)
        try:
            clip = next(clips)
        except StopIteration:
            raise RuntimeError(f"Video has fewer than {self.clip_size} sampled frames: {path}")
        features = self.extractor.extract_clip(clip.frames)
        return features, torch.tensor(label, dtype=torch.long)

def discover_records(dataset_path: Path):
    classes = sorted([p.name for p in dataset_path.iterdir() if p.is_dir()])
    if not classes:
        raise RuntimeError(f"No class directories found under {dataset_path}")
    mapping = {name: i for i, name in enumerate(classes)}
    records = []
    for cls in classes:
        for p in sorted((dataset_path / cls).rglob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                records.append((p, mapping[cls]))
    if not records:
        raise RuntimeError(f"No supported video files found under {dataset_path}")
    return classes, records

def split_by_video(records, seed=42, train_ratio=0.8, val_ratio=0.1):
    # Each record is a source video clip in this implementation; no generated
    # clip record is duplicated across splits.
    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)
    n = len(records)
    a = int(n * train_ratio)
    b = a + int(n * val_ratio)
    return records[:a], records[a:b], records[b:]

def run_epoch(loader, model, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = total = correct = 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        if train: optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        if train:
            loss.backward(); optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item(); total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(config.DATASET_PATH))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    dataset = Path(args.dataset)
    classes, records = discover_records(dataset)
    train_records, val_records, test_records = split_by_video(records, seed=args.seed)
    if not train_records or not val_records:
        raise RuntimeError("Dataset is too small for train/validation split")
    split_dir = Path(config.ROOT_DIR) / "data" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for filename, split_records in (("train.csv", train_records), ("validation.csv", val_records), ("test.csv", test_records)):
        with (split_dir / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video_path", "label_index"])
            for path, label in split_records:
                writer.writerow([str(path), label])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = VideoClipDataset(train_records, {c:i for i,c in enumerate(classes)}, config.SAMPLE_RATE, config.CLIP_SIZE, device)
    val_ds = VideoClipDataset(val_records, {c:i for i,c in enumerate(classes)}, config.SAMPLE_RATE, config.CLIP_SIZE, device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = CNNLSTMClassifier(config.FEATURE_SIZE, config.LSTM_HIDDEN_SIZE, config.LSTM_LAYERS, len(classes), config.DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(train_loader, model, criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(val_loader, model, criterion, optimizer, device, False)
        print(f"Epoch {epoch}/{args.epochs} | train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}")
        if va_loss < best:
            best = va_loss
            config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict()}, config.MODEL_PATH)
            config.CLASS_NAMES_PATH.write_text(json.dumps(classes, indent=2), encoding="utf-8")
            config.MODEL_CONFIG_PATH.write_text(json.dumps({
                "class_names": classes, "num_classes": len(classes), "feature_size": config.FEATURE_SIZE,
                "clip_size": config.CLIP_SIZE, "lstm_hidden_size": config.LSTM_HIDDEN_SIZE,
                "lstm_layers": config.LSTM_LAYERS, "dropout": config.DROPOUT,
                "sample_rate": config.SAMPLE_RATE, "image_size": config.IMAGE_SIZE,
            }, indent=2), encoding="utf-8")
    print(f"Training complete. Test videos held out: {len(test_records)}")

if __name__ == "__main__": main()
