---
title: "AutoMLBudgetPreset"
description: "Defines compute budget presets for AutoML runs."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines compute budget presets for AutoML runs.

## For Beginners

Think of this like choosing a "speed vs quality" mode:

- `CI` is the fastest and is designed for pipelines.
- `Fast` is for quick local experimentation.
- `Standard` is the recommended default for most users.
- `Thorough` spends more time searching for the best result.

## How It Works

These presets provide industry-standard defaults for time/trial limits and evaluation rigor.
Users can start with a preset and optionally override specific limits via configuration options.

## Fields

| Field | Summary |
|:-----|:--------|
| `CI` | A very small, deterministic budget intended for CI pipelines. |
| `Fast` | A small budget intended for quick local experimentation. |
| `Standard` | A balanced budget intended as the default for most use cases. |
| `Thorough` | A larger budget intended for deeper searches and better model quality. |

