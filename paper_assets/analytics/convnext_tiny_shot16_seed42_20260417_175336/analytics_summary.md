# Analytics Summary

Run: `convnext_tiny_shot16_seed42_20260417_175336`
Model: ConvNeXt-Tiny

Top-1 accuracy: 0.838
Top-5 accuracy: 0.994
Macro F1: 0.832
True class in Top-5: 0.994
True class ranked 2-5: 0.156

Standard analytics:
- rank histogram when Top-5 predictions are available
- cumulative Top-k curve and margins when logits are available
- margin by true-class rank when logits and Top-5 predictions are available
- feature-space geometry when frozen features are cached or provided
- prediction bias
- confusion matrix when available
- per-class performance when available