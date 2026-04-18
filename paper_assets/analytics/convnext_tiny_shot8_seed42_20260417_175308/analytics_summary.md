# Analytics Summary

Run: `convnext_tiny_shot8_seed42_20260417_175308`
Model: ConvNeXt-Tiny

Top-1 accuracy: 0.840
Top-5 accuracy: 0.996
Macro F1: 0.830
True class in Top-5: 0.996
True class ranked 2-5: 0.156

Standard analytics:
- rank histogram when Top-5 predictions are available
- cumulative Top-k curve and margins when logits are available
- margin by true-class rank when logits and Top-5 predictions are available
- feature-space geometry when frozen features are cached or provided
- prediction bias
- confusion matrix when available
- per-class performance when available