"""CNN feature sequence -> LSTM -> activity classifier."""
import torch
from torch import nn

class CNNLSTMClassifier(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_layers=1, num_classes=2, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (batch, sequence, features), got {tuple(x.shape)}")
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])
