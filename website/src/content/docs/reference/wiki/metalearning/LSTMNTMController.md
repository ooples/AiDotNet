---
title: "LSTMNTMController<T, TInput, TOutput>"
description: "LSTM-based NTM controller implementation with learnable parameters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

LSTM-based NTM controller implementation with learnable parameters.

## How It Works

This controller uses an LSTM cell to process inputs and generate addressing
parameters for the NTM memory. The LSTM maintains hidden state across timesteps,
enabling sequential reasoning and temporal dependencies.

**Architecture:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LSTMNTMController(NTMOptions<,,>)` | Initializes a new instance of LSTMNTMController with learnable weights. |

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

