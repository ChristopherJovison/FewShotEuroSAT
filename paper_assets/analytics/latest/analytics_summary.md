# Analytics Summary

Run: `latest`
Model: RemoteCLIP ViT-B/32 + GDA

Top-1 accuracy: 0.902
Top-5 accuracy: 0.996
Macro F1: 0.899
True class in Top-5: 0.996
True class ranked 2-5: 0.094

Standard analytics:
- rank histogram when Top-5 predictions are available
- cumulative Top-k curve and margins when logits are available
- margin by true-class rank when logits and Top-5 predictions are available
- feature-space geometry when frozen features are cached or provided
- prediction bias
- confusion matrix when available
- per-class performance when available