---
title: "TensorTapeOps"
description: "Tape-tracked element-wise tensor operations that wrap engine ops which otherwise bypass the autodiff graph."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Tape-tracked element-wise tensor operations that wrap engine ops which
otherwise bypass the autodiff graph.

## How It Works

`Tensor{` and
`Tensor{` are *not* recorded on the
autodiff tape — using them inside a layer's `Forward` path leaves the
downstream trainable parameters with zero gradients under tape-based training
(the LayerTestBase TapeGradient assertion specifically names this as a common
cause). The wrappers here construct a constant tensor of the operand's shape
and route the operation through `Tensor{` /
`Tensor{`, both of which are tape-connected; the
constant tensor itself is not a trainable leaf, so backward only propagates
gradients through the original tensor as the math demands.

Cost: one fresh constant tensor allocation per call. For hot inner loops with
fixed scalar+shape pairs, lift the tensor with `CreateDefault`
outside the loop and use `Tensor{` /
`Tensor{` directly. For one-off element-wise scalar
ops in `Forward`, these wrappers are the simplest correct replacement.

## Methods

| Method | Summary |
|:-----|:--------|
| `TapeAddScalar(IEngine,Tensor<>,)` | Tape-tracked replacement for `Tensor{`: returns `tensor + scalar` with the autodiff tape recording the op. |
| `TapeMultiplyScalar(IEngine,Tensor<>,)` | Tape-tracked replacement for `Tensor{`: returns `tensor * scalar` with the autodiff tape recording the op. |

