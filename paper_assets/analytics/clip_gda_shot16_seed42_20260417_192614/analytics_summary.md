# Analytics Summary

Run: `clip_gda_shot16_seed42_20260417_192614`
Model: CLIP ViT-B/16-QuickGELU + GDA

Top-1 accuracy: 0.894
Top-5 accuracy: 0.992
Macro F1: 0.893
True class in Top-5: 0.992
True class ranked 2-5: 0.098

Standard analytics:
- rank histogram when Top-5 predictions are available
- cumulative Top-k curve and margins when logits are available
- margin by true-class rank when logits and Top-5 predictions are available
- feature-space geometry when frozen features are cached or provided
- prediction bias
- confusion matrix when available
- per-class performance when available