---
title: "ActiveTransFSLOptions<T, TInput, TOutput>"
description: "Configuration options for ActiveTransFSL (Active Transductive Few-Shot Learning)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for ActiveTransFSL (Active Transductive Few-Shot Learning).

## How It Works

ActiveTransFSL combines active learning with transductive inference. After initial
adaptation on support data, it uses gradient-norm-based uncertainty to identify the most
uncertain parameter dimensions, then performs transductive refinement steps that focus
adaptation on these uncertain regions using query data feedback.

## Properties

| Property | Summary |
|:-----|:--------|
| `SelectionFraction` | Fraction of parameters (by uncertainty) to actively refine. |
| `TransductiveLR` | Learning rate for transductive refinement. |
| `TransductiveRefinementSteps` | Number of transductive refinement steps using query gradients. |
| `TransductiveWeight` | Weight on the transductive (query-feedback) loss. |

