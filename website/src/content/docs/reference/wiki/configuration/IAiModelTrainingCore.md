---
title: "IAiModelTrainingCore<T, TInput, TOutput>"
description: "Component that owns the core training configuration for an AI model build: the model itself, optimizer, regularization, fitness calculator, fit detector, training pipeline, training monitor, checkpoint manager, and memory management."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Configuration`

Component that owns the core training configuration for an AI model build: the model itself,
optimizer, regularization, fitness calculator, fit detector, training pipeline, training
monitor, checkpoint manager, and memory management. Extracted from `AiModelBuilder` as
slice 2 of the audit-2026-05 phase-2a DI refactor (see
`docs/internal/audit-2026-05-phase2a-aimodelbuilder-refactor.md`).

## How It Works

This concern is the dependency root for slices 3 / 4 / 6 / 7 / 9 / 10 / 11 per the migration
plan: cross-validation, compliance evaluation, workflow orchestration (FL / distributed),
advanced learning, storage, observability, and agent / export each consume the trained model
or the optimizer in some way, so they all wait for this component to land before they can
migrate.

## Properties

| Property | Summary |
|:-----|:--------|
| `CheckpointManager` | The configured checkpoint manager, or `null` if not configured (no checkpointing). |
| `FitDetector` | The configured fit detector (over- / under-fitting diagnostic), or `null` if not configured. |
| `FitnessCalculator` | The configured fitness calculator, or `null` if not configured. |
| `MemoryConfig` | The configured training-memory configuration (gradient checkpointing, activation pooling, sharding), or `null` for default settings. |
| `Model` | The configured model, or `null` if `ConfigureModel` hasn't been called. |
| `Optimizer` | The configured optimizer, or `null` if `ConfigureOptimizer` hasn't been called. |
| `Regularization` | The configured regularization strategy, or `null` if not configured. |
| `TrainingMonitor` | The configured training monitor, or `null` if not configured (no monitoring callbacks). |
| `TrainingPipelineConfiguration` | The configured training-pipeline configuration (custom stage definitions), or `null` for the default linear training pipeline. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ConfigureCheckpointManager(ICheckpointManager<,,>)` | Sets the checkpoint manager. |
| `ConfigureFitDetector(IFitDetector<,,>)` | Sets the fit detector. |
| `ConfigureFitnessCalculator(IFitnessCalculator<,,>)` | Sets the fitness calculator. |
| `ConfigureMemoryManagement(TrainingMemoryConfig)` | Sets the training-memory configuration. |
| `ConfigureModel(IFullModel<,,>)` | Sets the model. |
| `ConfigureOptimizer(IOptimizer<,,>)` | Sets the optimizer. |
| `ConfigureRegularization(IRegularization<,,>)` | Sets the regularization strategy. |
| `ConfigureTrainingMonitor(ITrainingMonitor<>)` | Sets the training monitor. |
| `ConfigureTrainingPipeline(TrainingPipelineConfiguration<,,>)` | Sets the training-pipeline configuration. |

