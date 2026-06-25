---
title: "MLPNTMController<T, TInput, TOutput>"
description: "MLP-based NTM controller implementation with learnable parameters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

MLP-based NTM controller implementation with learnable parameters.

## How It Works

This controller uses a multi-layer perceptron to process inputs and generate
addressing parameters for the NTM memory. Unlike LSTM, MLP is stateless and
processes each timestep independently.

**Architecture:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MLPNTMController(NTMOptions<,,>)` | Initializes a new instance of MLPNTMController with learnable weights. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>,List<Tensor<>>)` |  |
| `GenerateAddVector(Tensor<>)` |  |
| `GenerateEraseVector(Tensor<>)` |  |
| `GenerateOutput(Tensor<>,List<Tensor<>>)` |  |
| `GenerateReadKeys(Tensor<>)` |  |
| `GenerateWriteKey(Tensor<>)` |  |
| `GetParameters` |  |
| `Reset` |  |
| `SetParameters(Vector<>)` |  |

