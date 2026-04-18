# Analytics Summary

Run: `clip_gda_shot2_seed42_20260417_175143`
Model: CLIP ViT-B/16 + GDA

Top-1 accuracy: 0.606
Top-5 accuracy: 0.934
Macro F1: 0.583
True class in Top-5: 0.934
True class ranked 2-5: 0.328

Standard analytics:
- rank histogram when Top-5 predictions are available
- cumulative Top-k curve and margins when logits are available
- margin by true-class rank when logits and Top-5 predictions are available
- feature-space geometry when frozen features are cached or provided
- prediction bias
- confusion matrix when available
- per-class performance when available