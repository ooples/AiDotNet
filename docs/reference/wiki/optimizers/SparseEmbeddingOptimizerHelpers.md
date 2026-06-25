---
title: "SparseEmbeddingOptimizerHelpers"
description: "Sparse scatter helper for AdaDelta (Zeiler, 2012)."
section: "Reference"
---

_Optimizers_

Sparse scatter helper for AdaDelta (Zeiler, 2012). Maintains EMA of
squared gradients (accumGrad) and EMA of squared updates (accumDelta);
each step computes `dx = sqrt(accumDelta + ε) / sqrt(accumGrad + ε) · g`
and updates both accumulators.

## How It Works

**Strategy.** Touched indices are grouped by block (block-id =
`idx / blockSize`). For each touched block:

Dequantize the full block into a transient FP64 buffer.Apply the Adam moment + parameter update at the touched indices
only — untouched mDeq / vDeq stay at their dequantized values.Compute the block's new max-abs from the post-update mDeq / vDeq.Re-quantize the full block with the new scales.

This makes the cost `O(blockSize · touchedBlocks)` rather than
`O(paramLen)`. For paper-default LayoutXLM (vocab=250 002, dim=768,
~16 touched rows, blockSize=4096) that's ~12 blocks × 4 096 = 49 k
element-ops vs ~192 M dense — a ~4000× reduction.

**Config eligibility.** The helper handles the most common
configuration: `CompressBothMoments=true`, `QuantizationPercentile≥100`
(absolute-max scale), no stochastic rounding. Falls back to the
ToDense path for the other configurations so quantization semantics
stay bit-identical.

