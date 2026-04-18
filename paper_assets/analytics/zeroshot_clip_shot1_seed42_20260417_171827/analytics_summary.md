# Analytics Summary

Run: `zeroshot_clip_shot1_seed42_20260417_171827`
Model: Zero-shot CLIP ViT-B/16

Top-1 accuracy: 0.374
Top-5 accuracy: 0.878
Macro F1: 0.359
True class in Top-5: 0.878
True class ranked 2-5: 0.504

Standard analytics:
- rank histogram when Top-5 predictions are available
- cumulative Top-k curve and margins when logits are available
- margin by true-class rank when logits and Top-5 predictions are available
- feature-space geometry when frozen features are cached or provided
- prediction bias
- confusion matrix when available
- per-class performance when available