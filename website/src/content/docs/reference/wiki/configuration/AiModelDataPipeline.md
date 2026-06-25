---
title: "AiModelDataPipeline<T, TInput, TOutput>"
description: "Default implementation of `IAiModelDataPipeline`."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Configuration`

Default implementation of `IAiModelDataPipeline`. Mirrors the
preprocessing / postprocessing / data-loading / data-preparation / augmentation logic that
previously lived inline in `AiModelBuilder`, with no behavioural change. The facade
continues to be the supported entry point; this class is the audit-2026-05 phase-2a internal
reorganisation that makes the data-pipeline surface testable in isolation and replaceable
without touching the god class.

## Properties

| Property | Summary |
|:-----|:--------|
| `AugmentationConfig` |  |
| `DataLoader` |  |
| `DataPreparationPipeline` |  |
| `PostprocessingFitMaxRows` |  |
| `PostprocessingPipeline` |  |
| `PreprocessingPipeline` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ConfigureAugmentation(AugmentationConfig)` |  |
| `ConfigureAugmentation(AugmentationConfig<,>)` |  |
| `ConfigureDataLoader(IDataLoader<>)` |  |
| `ConfigureDataPreparation(Action<DataPreparationPipeline<>>)` |  |
| `ConfigurePostprocessing(Action<PostprocessingPipeline<,,>>)` |  |
| `ConfigurePostprocessing(IDataTransformer<,,>)` |  |
| `ConfigurePostprocessing(PostprocessingPipeline<,,>)` |  |
| `ConfigurePreprocessing(Action<PreprocessingPipeline<,,>>)` |  |
| `ConfigurePreprocessing(IDataTransformer<,,>)` |  |
| `ConfigurePreprocessing(PreprocessingPipeline<,,>)` |  |
| `SetPostprocessingFitMaxRows(Nullable<Int32>)` |  |

