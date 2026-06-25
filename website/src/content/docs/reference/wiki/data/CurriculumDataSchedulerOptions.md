---
title: "CurriculumDataSchedulerOptions"
description: "Configuration options for curriculum data scheduling."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Quality`

Configuration options for curriculum data scheduling.

## How It Works

Curriculum learning presents training samples in a meaningful order (easy to hard),
which can improve convergence speed and final model quality.

## Properties

| Property | Summary |
|:-----|:--------|
| `FullDataEpoch` | Epoch at which the full dataset becomes available. |
| `InitialFraction` | Initial fraction of the dataset available at the start. |
| `Order` | Curriculum ordering strategy. |
| `Pacing` | Pacing function controlling how fast harder samples are introduced. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

