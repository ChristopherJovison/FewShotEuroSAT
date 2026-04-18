# Analytics Summary

Run: `clip_gda_shot8_seed42_20260417_175253`
Model: CLIP ViT-B/16 + GDA

Top-1 accuracy: 0.801
Top-5 accuracy: 0.984
Macro F1: 0.792
True class in Top-5: 0.984
True class ranked 2-5: 0.183

Standard analytics:
- rank histogram when Top-5 predictions are available
- cumulative Top-k curve and margins when logits are available
- margin by true-class rank when logits and Top-5 predictions are available
- feature-space geometry when frozen features are cached or provided
- prediction bias
- confusion matrix when available
- per-class performance when available