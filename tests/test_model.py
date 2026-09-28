import torch
from src.models.cnn_lstm import CNNLSTMClassifier

def test_model_shape():
    m = CNNLSTMClassifier(input_size=512, hidden_size=32, num_layers=1, num_classes=5)
    y = m(torch.randn(2,16,512))
    assert y.shape == (2,5)
