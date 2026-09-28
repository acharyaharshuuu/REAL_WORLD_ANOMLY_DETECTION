# Viva Questions

1. Why ResNet18? — It provides pretrained spatial feature extraction with a compact 512-D representation.
2. Why LSTM? — The activity is a temporal sequence, so frame order and changes over time matter.
3. Why 16 frames? — It is the configured clip length in the project specification; the original notebook should remain the source of truth if it differs.
4. What does the CNN learn? — Spatial visual features within each frame.
5. What does the LSTM learn? — Temporal relationships across the frame-feature sequence.
6. Why not use Canny/grayscale as the final CNN input? — The final pretrained ResNet18 path expects RGB ImageNet-style preprocessing.
7. Why event generation? — To merge consecutive clip predictions into a more useful potential incident instead of producing repeated alerts.
8. Is a prediction a confirmed crime? — No. It is a model prediction requiring human review.
