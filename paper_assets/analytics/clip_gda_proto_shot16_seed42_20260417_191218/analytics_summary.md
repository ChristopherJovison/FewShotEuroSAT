# Analytics Summary

Run: `clip_gda_proto_shot16_seed42_20260417_191218`
Model: CLIP ViT-B/16 + GDA+Proto

Top-1 accuracy: 0.875
Top-5 accuracy: 0.994
Macro F1: 0.872
True class in Top-5: 0.994
True class ranked 2-5: 0.119

Standard analytics:
- rank histogram when Top-5 predictions are available
- cumulative Top-k curve and margins when logits are available
- margin by true-class rank when logits and Top-5 predictions are available
- feature-space geometry when frozen features are cached or provided
- prediction bias
- confusion matrix when available
- per-class performance when available