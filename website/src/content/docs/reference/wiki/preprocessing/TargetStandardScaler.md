---
title: "TargetStandardScaler<T, TOutput>"
description: "Standard (z-score) scaling for regression TARGETS — the `IDataTransformer` that `AiModelBuilder.ConfigureTargetScaling()` installs into the (previously always-null) `TargetPipeline`."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing`

Standard (z-score) scaling for regression TARGETS — the `IDataTransformer`
that `AiModelBuilder.ConfigureTargetScaling()` installs into the (previously always-null)
`TargetPipeline`.

## How It Works

**Why:** the facade has always scaled FEATURES (`ConfigurePreprocessing`) but never the
target — raw targets on wide scales (prices, dollar P&L) make gradient training diverge, so every
consumer hand-rolled a target scaler plus the inverse transform on the way out. The carrier
(`PreprocessingInfo.TargetPipeline` + `InverseTransformPredictions`) existed but nothing ever
populated or invoked it; this transformer + the builder/predict wiring completes the feature.

**Shapes:** targets are adapted to a column matrix for the underlying
`StandardScaler` — `Vector<T>` ↔ `Matrix[n,1]`; `Tensor<T>`
rank-1 ↔ `Matrix[n,1]`; `Tensor<T>` rank-2 `[n,k]` ↔ `Matrix[n,k]` (each
output column scaled independently). Other output types are rejected at construction.

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnIndices` |  |
| `IsFitted` |  |
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit()` |  |
| `FitTransform()` |  |
| `GetFeatureNamesOut(String[])` |  |
| `InverseTransform()` |  |
| `Transform()` |  |

