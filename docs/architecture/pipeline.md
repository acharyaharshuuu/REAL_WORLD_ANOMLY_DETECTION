# System Pipeline

Video → sampling → 16-frame clip → RGB/ImageNet preprocessing → ResNet18 feature extraction → 16×512 feature sequence → LSTM → activity probability → temporal smoothing/event aggregation → Potential Incident.

CNN asks: “What is present in this frame?”  
LSTM asks: “How is this activity changing across frames?”
