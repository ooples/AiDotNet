---
title: "DatasetDistillerOptions"
description: "Configuration options for dataset distillation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Quality`

Configuration options for dataset distillation.

## How It Works

Dataset distillation synthesizes a small set of examples that capture the essence
of the full training set. Training on the distilled set produces similar model quality
to training on the full dataset.

## Properties

| Property | Summary |
|:-----|:--------|
| `DistillLearningRate` | Learning rate for optimizing distilled samples. |
| `NumSteps` | Number of distillation optimization steps. |
| `SamplesPerClass` | Number of synthetic samples per class to generate. |
| `Seed` | Random seed for reproducibility. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

