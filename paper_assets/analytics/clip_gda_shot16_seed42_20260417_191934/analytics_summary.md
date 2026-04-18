# Analytics Summary

Run: `clip_gda_shot16_seed42_20260417_191934`
Model: CLIP ViT-B/16 + GDA

Top-1 accuracy: 0.883
Top-5 accuracy: 0.995
Macro F1: 0.879
True class in Top-5: 0.995
True class ranked 2-5: 0.113

Standard analytics:
- rank histogram when Top-5 predictions are available
- cumulative Top-k curve and margins when logits are available
- margin by true-class rank when logits and Top-5 predictions are available
- feature-space geometry when frozen features are cached or provided
- prediction bias
- confusion matrix when available
- per-class performance when available