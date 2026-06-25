---
title: "TrainerSettings"
description: "Configuration for the trainer behavior section of a training recipe."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Training.Configuration`

Configuration for the trainer behavior section of a training recipe.

## For Beginners

These settings control how the training loop runs - how many
times it goes through the data (epochs), whether to print progress, and an optional
random seed for reproducible results.

## Properties

| Property | Summary |
|:-----|:--------|
| `EnableLogging` | Gets or sets whether to log training progress (epoch number and loss). |
| `Epochs` | Gets or sets the number of training epochs (full passes through the data). |
| `Seed` | Gets or sets an optional random seed for reproducible training runs. |

