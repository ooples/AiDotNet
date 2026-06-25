---
title: "RobustnessEngine<T>"
description: "Engine for analyzing model robustness to input perturbations and noise."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Engines`

Engine for analyzing model robustness to input perturbations and noise.

## For Beginners

Robustness analysis tests how well your model handles:

- **Noise:** Random perturbations to input features
- **Missing data:** Features replaced with default values
- **Outliers:** Extreme values in the input
- **Distribution shift:** Data that differs from training

## How It Works

**Why robustness matters:**

- Real-world data is noisy and imperfect
- Small changes shouldn't cause dramatic prediction changes
- Models should degrade gracefully, not catastrophically

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RobustnessEngine(RobustnessOptions)` | Initializes the robustness engine. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Analyze([0:,0:],[],Func<,[0:,0:],[]>,,String,Boolean,Boolean)` | Analyzes model robustness by testing with various perturbations. |

