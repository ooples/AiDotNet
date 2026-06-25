---
title: "LoadResult"
description: "Result of a model weight loading operation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelLoading`

Result of a model weight loading operation.

## Properties

| Property | Summary |
|:-----|:--------|
| `ErrorMessage` | Error message if loading failed. |
| `MappedNames` | List of mapped tensor name pairs (source, target). |
| `MappedTensors` | Number of tensors successfully mapped. |
| `Success` | Whether the loading was successful. |
| `TotalTensors` | Total number of tensors in the file. |
| `UnmappedNames` | List of unmapped tensor names. |
| `UnmappedTensors` | Number of tensors that couldn't be mapped. |
| `WeightsApplied` | Whether weights were actually applied to the model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Gets a summary of the load result. |

