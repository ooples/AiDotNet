---
title: "DREAMOptions<T, TInput, TOutput>"
description: "Configuration options for DREAM: Directed REward Augmented Meta-learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for DREAM: Directed REward Augmented Meta-learning.

## How It Works

DREAM augments MAML-style meta-learning with a learned reward/loss shaping function
that transforms the raw task loss into a more informative gradient signal. The reward
shaper maps (loss, gradient_norm, step) → shaped_loss, enabling the inner loop
to receive curriculum-like guidance that accelerates adaptation.

## Properties

| Property | Summary |
|:-----|:--------|
| `RewardShaperHiddenDim` | Hidden dimension for the reward shaper MLP. |
| `RewardShapingWeight` | Weight for the reward shaping term in the total loss. |
| `ShapingDiscount` | Discount factor for shaped reward across adaptation steps. |

