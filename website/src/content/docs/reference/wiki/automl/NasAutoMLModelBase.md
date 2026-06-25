---
title: "NasAutoMLModelBase<T>"
description: "Base class for NAS-based AutoML models."
section: "API Reference"
---

`Base Classes` · `AiDotNet.AutoML.NAS`

Base class for NAS-based AutoML models.

## Properties

| Property | Summary |
|:-----|:--------|
| `BestArchitecture` | Gets the best architecture found by the NAS search. |
| `NasNumNodes` | Gets the number of nodes to search over. |
| `NasSearchSpace` | Gets the NAS search space. |
| `NumOps` | Gets the numeric operations provider for `T`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateModelAsync(Type,Dictionary<String,Object>)` | NAS creates models via `CancellationToken)` and `Architecture{`. |
| `GetDefaultSearchSpace(Type)` | NAS uses an architecture search space (`NasSearchSpace`) instead of traditional hyperparameter ranges. |
| `SearchArchitecture(Tensor<>,Tensor<>,Tensor<>,Tensor<>,TimeSpan,CancellationToken)` | Performs algorithm-specific architecture search. |
| `SearchAsync(Tensor<>,Tensor<>,Tensor<>,Tensor<>,TimeSpan,CancellationToken)` | Runs the NAS search and returns the best model found. |
| `SuggestNextTrialAsync` | NAS does not use traditional trial suggestion; architecture search is handled by `CancellationToken)`. |

