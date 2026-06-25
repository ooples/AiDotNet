---
title: "NTMModel<T, TInput, TOutput>"
description: "NTM model for inference with persistent memory."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

NTM model for inference with persistent memory.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NTMModel(INTMController<>,NTMMemory<>,List<NTMReadHead<>>,NTMWriteHead<>,NTMOptions<,,>)` | Initializes a new instance of the NTMModel class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Metadata` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CombineInputWithReadContents(Tensor<>)` | Combines input with previous read contents. |
| `ConvertInputToTensor()` | Converts input to tensor format. |
| `ConvertTensorToOutput(Tensor<>)` | Converts tensor output to the expected output type. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `Predict()` |  |
| `ProcessTimestepInternal(Tensor<>)` | Processes a single timestep using the model's internal components. |
| `Train(,)` |  |
| `UpdateParameters(Vector<>)` |  |

