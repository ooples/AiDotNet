---
title: "MathDataLoader<T>"
description: "Loads the Hendrycks MATH benchmark — competition math problems (Hendrycks et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads the Hendrycks MATH benchmark — competition math problems (Hendrycks et al. 2021).

## How It Works

Expects:

where `subject` is one of: algebra, counting_and_probability,
geometry, intermediate_algebra, number_theory, prealgebra, precalculus.
Each JSON has fields `problem`, `level`, `type`, `solution`.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

