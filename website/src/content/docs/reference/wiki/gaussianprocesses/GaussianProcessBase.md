---
title: "GaussianProcessBase<T>"
description: "Abstract base class for Gaussian Process models that provides IFullModel compliance."
section: "API Reference"
---

`Base Classes` · `AiDotNet.GaussianProcesses`

Abstract base class for Gaussian Process models that provides IFullModel compliance.
Maps the GP Fit/Predict(single) API to the standard Train/Predict(batch) API.

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DeepCopy` |  |
| `Fit(Matrix<>,Vector<>)` |  |
| `GetParameters` |  |
| `Predict(Matrix<>)` | Predicts mean values for each row of the input matrix. |
| `Predict(Vector<>)` |  |
| `SetParameters(Vector<>)` |  |
| `Train(Matrix<>,Vector<>)` | Trains the GP. |
| `UpdateKernel(IKernelFunction<>)` |  |
| `WithParameters(Vector<>)` |  |

