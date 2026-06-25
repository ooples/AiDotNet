---
title: "SequenceLastLayer<T>"
description: "A layer that extracts the last timestep from a sequence."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

A layer that extracts the last timestep from a sequence.

## For Beginners

When processing sequences (like sentences or time series),
recurrent layers output a value for each timestep. For tasks like classification,
we often only need the final output (after seeing the whole sequence). This layer
extracts just that last output.

## How It Works

This layer is used after recurrent layers (RNN, LSTM, GRU) when the task requires
a single output from the entire sequence, such as sequence classification.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SequenceLastLayer(Int32)` | Initializes a new SequenceLastLayer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsGpuExecution` | Indicates whether this layer supports GPU execution. |
| `SupportsGpuTraining` |  |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Extracts the last timestep from the input sequence. |
| `ForwardGpu(Tensor<>[])` | GPU-accelerated forward pass that extracts the last timestep from a sequence. |
| `GetParameters` | Returns an empty vector since this layer has no trainable parameters. |
| `ResetState` | Reset state is a no-op since this layer maintains no state between forward passes. |
| `UpdateParameters()` | Update parameters is a no-op since this layer has no trainable parameters. |

