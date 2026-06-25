---
title: "IAiModelDataPipeline<T, TInput, TOutput>"
description: "Component that owns the data-pipeline configuration for an AI model build: preprocessing, postprocessing, data loading, data preparation, and augmentation."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Configuration`

Component that owns the data-pipeline configuration for an AI model build: preprocessing,
postprocessing, data loading, data preparation, and augmentation. Extracted from
`AiModelBuilder` as slice 1 of the audit-2026-05 phase-2a DI refactor (see
`docs/internal/audit-2026-05-phase2a-aimodelbuilder-refactor.md`).

## How It Works

The interface exposes the configured state via get-only properties so that the rest of the
build pipeline (`AiModelBuilder.BuildAsync`, the partial-class siblings, and any
alternative composition root such as a future YAML loader) can consume it without depending
on the god class. The `Configure*` methods mutate the underlying component instance and
return `void` — the fluent chaining returns happen at the `AiModelBuilder` facade
layer, not here, so the component stays usable from non-fluent contexts (e.g. tests).

This is the public extension seam: third-party packages or downstream consumers can implement
`IAiModelDataPipeline` with custom defaults (e.g. a domain-
specific preprocessing pipeline that ships with the audit-2026-05 federal-use SDK) and inject
it into `AiModelBuilder` via the ctor parameter added in the same slice.

## Properties

| Property | Summary |
|:-----|:--------|
| `AugmentationConfig` | The configured (non-row-changing) augmentation configuration, or `null` if `ConfigureAugmentation` hasn't been called. |
| `DataLoader` | The configured data loader, or `null` if `ConfigureDataLoader` hasn't been called. |
| `DataPreparationPipeline` | The configured row-changing data-preparation pipeline (outlier removal, SMOTE augmentation, etc.), or `null` if `ConfigureDataPreparation` hasn't been called. |
| `PostprocessingFitMaxRows` | Optional cap on the number of training rows fed into the post-train pipeline-fit `Predict` call. |
| `PostprocessingPipeline` | The configured postprocessing pipeline, or `null` if `ConfigurePostprocessing` hasn't been called. |
| `PreprocessingPipeline` | The configured preprocessing pipeline, or `null` if `ConfigurePreprocessing` hasn't been called. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ConfigureAugmentation(AugmentationConfig)` | Configures augmentation with an explicit (possibly null) config. |
| `ConfigureAugmentation(AugmentationConfig<,>)` | Strongly-typed overload of `AugmentationConfig)` with IDE-discoverable typed `Augmenter` slot. |
| `ConfigureDataLoader(IDataLoader<>)` | Configures the data loader. |
| `ConfigureDataPreparation(Action<DataPreparationPipeline<>>)` | Configures the row-changing data-preparation pipeline. |
| `ConfigurePostprocessing(Action<PostprocessingPipeline<,,>>)` | Configures postprocessing using a pipeline-builder callback. |
| `ConfigurePostprocessing(IDataTransformer<,,>)` | Configures postprocessing with a single transformer. |
| `ConfigurePostprocessing(PostprocessingPipeline<,,>)` | Configures postprocessing from a pre-built pipeline. |
| `ConfigurePreprocessing(Action<PreprocessingPipeline<,,>>)` | Configures preprocessing using a pipeline-builder callback. |
| `ConfigurePreprocessing(IDataTransformer<,,>)` | Configures preprocessing with a single transformer. |
| `ConfigurePreprocessing(PreprocessingPipeline<,,>)` | Configures preprocessing from a pre-built pipeline. |
| `SetPostprocessingFitMaxRows(Nullable<Int32>)` | Caps the number of training rows fed into the post-train pipeline-fit `Predict`. |

