---
title: "RecurrentHyperNetOptions<T, TInput, TOutput>"
description: "Configuration options for Recurrent HyperNetwork meta-learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Recurrent HyperNetwork meta-learning.

## How It Works

A GRU-like recurrent cell processes gradient information at each adaptation step,
maintaining hidden state that captures the optimization trajectory. The recurrent
output modulates per-parameter learning rates, enabling adaptive step sizes that
evolve through the inner loop.

## Properties

| Property | Summary |
|:-----|:--------|
| `CellRegWeight` | L2 regularization weight on recurrent cell state magnitude to prevent unbounded hidden state growth. |
| `ForgetBias` | Initial forget gate bias. |
| `HiddenStateDim` | Dimension of the GRU hidden state. |
| `InputDim` | Gradient compression dimension for feeding into the recurrent cell. |

