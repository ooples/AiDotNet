---
title: "TrainingRecipeConfig"
description: "Root configuration object for a complete training recipe defined in YAML."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Training.Configuration`

Root configuration object for a complete training recipe defined in YAML.

## For Beginners

A training recipe brings together all the pieces needed to train a model:
the model architecture, the dataset, the optimizer, the loss function, and training settings.
You can define all of this in a single YAML file and load it with `YamlConfigLoader`.

## How It Works

**Example YAML:**

## Properties

| Property | Summary |
|:-----|:--------|
| `Dataset` | Gets or sets the dataset configuration section. |
| `LossFunction` | Gets or sets the loss function configuration section. |
| `Model` | Gets or sets the model configuration section. |
| `Optimizer` | Gets or sets the optimizer configuration section. |
| `Trainer` | Gets or sets the trainer settings section. |

