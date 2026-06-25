---
title: "FederatedCompressionStrategy"
description: "Specifies the compression strategy for federated model updates."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the compression strategy for federated model updates.

## For Beginners

Each strategy determines how client updates are compressed before
sending to the server, trading off accuracy for communication efficiency.

## Fields

| Field | Summary |
|:-----|:--------|
| `Advanced` | Advanced — use the advanced compression options (PowerSGD, sketching, etc.). |
| `None` | No compression — send full model updates. |
| `RandomK` | Random-K sparsification — send K randomly selected gradient elements. |
| `StochasticQuantization` | Stochastic quantization — probabilistically round to fewer bits (unbiased). |
| `Threshold` | Threshold sparsification — send only elements exceeding an absolute threshold. |
| `TopK` | Top-K sparsification — send only the K largest gradient elements by magnitude. |
| `UniformQuantization` | Uniform quantization — deterministically reduce precision to fewer bits. |

