---
title: "FineTuningConfiguration<T, TInput, TOutput>"
description: "Configuration for fine-tuning during model building."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration for fine-tuning during model building.

## For Beginners

This is the fine-tuning "on/off switch" and settings bundle.
You can leave it alone to skip fine-tuning, or configure it to apply preference learning,
RLHF, or other alignment techniques after initial training.

## How It Works

This configuration controls whether fine-tuning is enabled and which method/implementation is used.
When enabled and no custom implementation is provided, a default fine-tuning method is created
based on the configured options.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoSplitForValidation` | Gets or sets whether to auto-split training data for validation. |
| `Enabled` | Gets or sets whether fine-tuning is enabled. |
| `Implementation` | Gets or sets an optional custom fine-tuning implementation. |
| `Options` | Gets or sets the fine-tuning options. |
| `TrainingData` | Gets or sets the training data for fine-tuning. |
| `ValidationData` | Gets or sets optional validation data for fine-tuning. |
| `ValidationSplitRatio` | Gets or sets the ratio of data to use for validation if ValidationData is not provided. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForDPO(FineTuningData<,,>)` | Creates a minimal configuration for DPO with default options. |
| `ForGRPO(FineTuningData<,,>)` | Creates a minimal configuration for GRPO with default options. |
| `ForORPO(FineTuningData<,,>)` | Creates a minimal configuration for ORPO with default options. |
| `ForSFT(FineTuningData<,,>)` | Creates a minimal configuration for SFT with default options. |
| `ForSimPO(FineTuningData<,,>)` | Creates a minimal configuration for SimPO with default options. |

