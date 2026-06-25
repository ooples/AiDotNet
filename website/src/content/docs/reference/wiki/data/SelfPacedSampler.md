---
title: "SelfPacedSampler<T>"
description: "A sampler that implements self-paced learning with automatic difficulty adjustment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Sampling`

A sampler that implements self-paced learning with automatic difficulty adjustment.

## For Beginners

Unlike CurriculumSampler where YOU define difficulty,
SelfPacedSampler lets the MODEL decide what's easy based on its own performance:

- Samples with low loss = easy = selected early
- Samples with high loss = hard = selected later

This is adaptive curriculum learning - the curriculum adjusts based on the model!

## How It Works

SelfPacedSampler automatically adjusts sample selection based on the model's
performance on each sample. Samples with lower loss (easier for the model)
are more likely to be selected early, with harder samples gradually introduced.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelfPacedSampler(Int32,,,Int32,Nullable<Int32>)` | Initializes a new instance of the SelfPacedSampler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Lambda` | Gets the current pace parameter lambda. |
| `Length` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetIndicesCore` |  |
| `OnEpochStartCore(Int32)` |  |
| `UpdateLoss(Int32,)` | Updates the loss for a specific sample. |
| `UpdateLosses(IReadOnlyList<Int32>,IReadOnlyList<>)` | Batch updates losses for multiple samples. |

