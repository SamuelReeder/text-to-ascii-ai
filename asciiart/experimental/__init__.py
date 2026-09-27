"""Approaches that were tried and measured but are not part of the final converter.

* model.py / scripts/experimental/train_asciinet.py: a ~10M-param CNN trained from scratch on
  98k images to pick characters by rendering glyphs differentiably (straight-through Gumbel)
  and matching the image at several blur scales. It reached 0.22 CLIP image-retrieval@1 vs 0.33
  for plain per-cell glyph matching.
* refine.py: exact coordinate descent on that multi-scale render loss (23% lower loss than
  per-cell matching, but no legibility gain: 0.30-0.32 vs 0.33).
* guided.py: glyph search guided by SigLIP gradients: no gain on a held-out judge.
* synthetic.py: procedural text/shape/line-art training images.
"""
