---
title: "ValidationHelper<T>"
description: "Provides validation methods for AI model inputs and parameters."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides validation methods for AI model inputs and parameters.

## How It Works

**For Beginners:** This helper class ensures that the data you provide to AI models is valid and properly formatted.
It can handle both traditional matrix/vector inputs (for regression-like models) and tensor inputs (for neural networks).
Think of it as a quality control checkpoint that prevents errors before they happen by checking that your
data meets all the requirements needed for successful model training and prediction.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetCallerInfo(Int32)` | Gets information about the calling method. |
| `ResolveCallerInfo(String,String,Int32)` | Resolves component and operation names, using caller info if either is empty. |
| `ValidateInputData(,)` | Validates that input data is properly formatted for model training. |
| `ValidateInputData(OptimizationInputData<,,>)` | Validates that optimization input data is properly formatted for model training and evaluation. |
| `ValidatePoissonData(Vector<>)` | Validates that data is appropriate for Poisson regression. |

