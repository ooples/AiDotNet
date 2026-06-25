---
title: "NF4Quantizer<T, TInput, TOutput>"
description: "NF4 (4-bit NormalFloat) quantizer - optimal for normally distributed weights."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Optimization.Quantization.Formats`

NF4 (4-bit NormalFloat) quantizer - optimal for normally distributed weights.
Used by QLoRA for efficient 4-bit base model quantization.

## For Beginners

NF4 is a special 4-bit format where the 16 possible values
are chosen to be optimal for weights that follow a normal distribution (bell curve).
This makes it perfect for neural network weights, which are typically normally distributed.

## How It Works

**How It Works:**

**Key Features:**

**Reference:** Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NF4Quantizer(QuantizationConfiguration,Int32)` | Initializes a new instance of the NF4Quantizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BitWidth` |  |
| `IsCalibrated` | Gets whether the quantizer has been calibrated. |
| `Mode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calibrate(IFullModel<,,>,IEnumerable<>)` |  |
| `FindNearestNF4Value(Double)` | Finds the nearest value in the NF4 codebook. |
| `GetScaleFactor(String)` |  |
| `GetZeroPoint(String)` |  |
| `IndexToNF4(Int32)` | Converts a 4-bit index to its NF4 value. |
| `NF4ToIndex(Double)` | Converts a value to its nearest 4-bit index. |
| `Quantize(IFullModel<,,>,QuantizationConfiguration)` |  |
| `QuantizeWithNF4(Vector<>)` | Quantizes parameters using NF4 format. |

