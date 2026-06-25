---
title: "DataLoaderCheckpoint"
description: "Serializable checkpoint for a stateful data loader."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Serializable checkpoint for a stateful data loader.

## How It Works

Contains all information needed to resume data iteration from an exact position,
including the current index, epoch, shuffle order, and RNG state.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | The batch size at checkpoint time. |
| `CreatedAtUtc` | UTC timestamp when the checkpoint was created. |
| `CurrentBatchIndex` | The current batch index within the epoch. |
| `CurrentIndex` | The current sample index within the epoch. |
| `Epoch` | The current epoch number. |
| `Metadata` | Optional metadata for custom state information. |
| `RandomSeed` | The random seed used to generate the current shuffle order. |
| `ShuffledIndices` | The shuffled indices for the current epoch, preserving iteration order. |
| `TotalCount` | The total number of samples in the dataset at checkpoint time. |

