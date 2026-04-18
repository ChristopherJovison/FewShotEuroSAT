# Analytics Summary

Run: `clip_tip_adapter_shot1_seed42_20260417_174838`
Model: CLIP ViT-B/16 + Tip-Adapter

Top-1 accuracy: 0.454
Top-5 accuracy: 0.906
Macro F1: 0.446
True class in Top-5: 0.906
True class ranked 2-5: 0.451

Standard analytics:
- rank histogram when Top-5 predictions are available
- cumulative Top-k curve and margins when logits are available
- margin by true-class rank when logits and Top-5 predictions are available
- feature-space geometry when frozen features are cached or provided
- prediction bias
- confusion matrix when available
- per-class performance when available