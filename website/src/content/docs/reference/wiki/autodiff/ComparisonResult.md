---
title: "ComparisonResult<T>"
description: "Result of comparing numerical and analytical gradients."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Autodiff.Testing`

Result of comparing numerical and analytical gradients.

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageRelativeError` | Average relative error. |
| `Errors` | Detailed error messages for failed elements. |
| `FailedElements` | Number of elements that failed verification. |
| `MaxRelativeError` | Maximum relative error observed. |
| `Passed` | Whether all gradients passed verification. |
| `TotalElementsChecked` | Total elements checked. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Returns a summary string of the comparison result. |

