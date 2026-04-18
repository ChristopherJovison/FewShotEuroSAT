# Analytics Summary

Run: `clip_tip_adapter_shot2_seed42_20260417_175152`
Model: CLIP ViT-B/16 + Tip-Adapter

Top-1 accuracy: 0.570
Top-5 accuracy: 0.914
Macro F1: 0.528
True class in Top-5: 0.914
True class ranked 2-5: 0.343

Standard analytics:
- rank histogram when Top-5 predictions are available
- cumulative Top-k curve and margins when logits are available
- margin by true-class rank when logits and Top-5 predictions are available
- feature-space geometry when frozen features are cached or provided
- prediction bias
- confusion matrix when available
- per-class performance when available