# Analytics Summary

Run: `zeroshot_clip_shot16_seed42_20260417_201526`
Model: Zero-shot RemoteCLIP ViT-B/32

Top-1 accuracy: 0.376
Top-5 accuracy: 0.940
Macro F1: 0.343
True class in Top-5: 0.940
True class ranked 2-5: 0.564

Standard analytics:
- rank histogram when Top-5 predictions are available
- cumulative Top-k curve and margins when logits are available
- margin by true-class rank when logits and Top-5 predictions are available
- feature-space geometry when frozen features are cached or provided
- prediction bias
- confusion matrix when available
- per-class performance when available