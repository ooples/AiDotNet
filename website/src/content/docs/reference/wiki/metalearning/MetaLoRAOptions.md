---
title: "MetaLoRAOptions<T, TInput, TOutput>"
description: "Configuration options for Meta-LoRA (Low-Rank Adaptation for Meta-Learning)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Meta-LoRA (Low-Rank Adaptation for Meta-Learning).

## How It Works

Meta-LoRA applies the Low-Rank Adaptation principle to meta-learning: instead of adapting
all model parameters during the inner loop (as in MAML), it meta-learns a set of low-rank
basis vectors and only adapts a small number of coefficients per task. This drastically
reduces the inner-loop parameter count from d to r (where r << d).

**Key Parameters:**

- `Rank` — number of low-rank basis vectors (controls capacity vs efficiency)
- `ScalingAlpha` — scales the LoRA update magnitude (analogous to alpha/r in standard LoRA)

## Properties

| Property | Summary |
|:-----|:--------|
| `BasisInitStdDev` | Standard deviation for initializing the low-rank basis vectors. |
| `Rank` | Number of low-rank basis vectors used for adaptation. |
| `ScalingAlpha` | Scaling factor applied to the LoRA update: adapted = base + (alpha / rank) * sum(c_i * v_i). |

