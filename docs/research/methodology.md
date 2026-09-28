# Methodology

The software layer implements the intended CNN+LSTM research pipeline without altering the existing notebooks. ResNet18 is used as a pretrained spatial feature extractor, producing a 512-dimensional vector per sampled frame. A 16-frame sequence is passed to an LSTM for temporal learning. Predictions are converted into reviewable events using confidence and consecutive-clip rules.

Notebook-specific values should be reconciled here after the original notebooks are available in the project repository.
