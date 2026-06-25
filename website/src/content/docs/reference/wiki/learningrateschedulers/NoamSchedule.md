---
title: "NoamSchedule"
description: "Implements the Noam learning rate schedule from \"Attention Is All You Need\" (Vaswani et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LearningRateSchedulers`

Implements the Noam learning rate schedule from "Attention Is All You Need"
(Vaswani et al. 2017, §5.3): inverse-square-root decay with linear warmup.

## For Beginners

Transformers are sensitive to learning-rate
choice early in training because attention weights are softmax-normalized
and gradients can explode when the network has to figure out which tokens
to attend to. The Noam schedule ramps up the learning rate slowly for the
first few thousand steps (the "warmup"), then decreases it like the
inverse square root of the step number. This is how every Transformer in
the original 2017 paper was trained.

## How It Works

Formula:

where t is the 1-indexed training step from the paper. Peaks at t = warmup
with value `factor · d_model^(-0.5) · warmup^(-0.5)`, then decays as
t^(-0.5).

Step-counter convention (matches PyTorch / HuggingFace transformer
schedulers): the library's `_currentStep`
is incremented at end-of-batch and represents "batches completed so far"
(0-based). The Noam paper's t is 1-based, so this scheduler maps
`t = step + 1` internally:

- Before any Step() call, `_currentStep = 0` ⇒ `t = 1` ⇒ warmup-start LR.
- Batch N reads the LR that was set by the (N-1)th Step() call ⇒ lr(t=N) ⇒ `t = step + 1` with `step = N-1`.
- Reset restores the warmup-start LR (NOT `_baseLearningRate`, which we use as a peak-LR sentinel for the base ctor's positive guard).

This schedule pairs with Adam β₁=0.9, β₂=0.98, ε=1e-9 (the Vaswani 2017
hyperparameters): the small β₂ tracks rapidly-changing attention/embedding
gradients, and the warmup phase keeps the initial updates small while the
second-moment estimates have not yet stabilized. Without warmup, β₂=0.98
is too aggressive for early-training stability — which is why these values
must be applied together, not piecewise.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NoamSchedule(Int32,Int32,Double)` | Initializes a new Noam schedule (Vaswani 2017 inverse-sqrt with linear warmup). |

## Properties

| Property | Summary |
|:-----|:--------|
| `Factor` | Multiplicative scale on the schedule (1.0 = paper-faithful). |
| `ModelDimension` | Model dimension this schedule was configured for. |
| `WarmupSteps` | Number of warmup steps configured. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLearningRate(Int32)` |  |
| `GetState` |  |
| `Reset` | Restores the scheduler to its initial state. |

