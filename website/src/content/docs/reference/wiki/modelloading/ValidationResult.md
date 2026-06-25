---
title: "ValidationResult"
description: "Result of weights file validation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelLoading`

Result of weights file validation.

## Properties

| Property | Summary |
|:-----|:--------|
| `ErrorMessage` | Error message if validation failed. |
| `FoundPatterns` | Patterns that were found. |
| `IsValid` | Whether the validation passed. |
| `MatchedTensors` | Tensor names that matched the patterns. |
| `MissingPatterns` | Patterns that were not found. |
| `Path` | Path to the validated file. |
| `TotalSizeBytes` | Total size of all tensors in bytes. |
| `TotalTensors` | Total number of tensors in the file. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Gets a summary of the validation result. |

