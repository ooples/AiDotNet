---
title: "AiModelTrainingCore<T, TInput, TOutput>"
description: "Default implementation of `IAiModelTrainingCore`."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Configuration`

Default implementation of `IAiModelTrainingCore`. Mirrors the
pre-refactor inline storage that `AiModelBuilder` used for its model / optimizer /
regularization / fitness / fit-detector / training-pipeline / checkpoint / memory / monitor
fields. Audit-2026-05 phase-2a slice 2 (see
`docs/internal/audit-2026-05-phase2a-aimodelbuilder-refactor.md`).

## Properties

| Property | Summary |
|:-----|:--------|
| `CheckpointManager` |  |
| `FitDetector` |  |
| `FitnessCalculator` |  |
| `MemoryConfig` |  |
| `Model` |  |
| `Optimizer` |  |
| `Regularization` |  |
| `TrainingMonitor` |  |
| `TrainingPipelineConfiguration` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ConfigureCheckpointManager(ICheckpointManager<,,>)` |  |
| `ConfigureFitDetector(IFitDetector<,,>)` |  |
| `ConfigureFitnessCalculator(IFitnessCalculator<,,>)` |  |
| `ConfigureMemoryManagement(TrainingMemoryConfig)` |  |
| `ConfigureModel(IFullModel<,,>)` |  |
| `ConfigureOptimizer(IOptimizer<,,>)` |  |
| `ConfigureRegularization(IRegularization<,,>)` |  |
| `ConfigureTrainingMonitor(ITrainingMonitor<>)` |  |
| `ConfigureTrainingPipeline(TrainingPipelineConfiguration<,,>)` |  |

