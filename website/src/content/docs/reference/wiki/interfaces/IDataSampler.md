---
title: "IDataSampler"
description: "Defines the contract for sampling indices from a dataset during batch iteration."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for sampling indices from a dataset during batch iteration.

## For Beginners

A sampler decides which data points to include in each batch
and in what order. The default is random sampling, but you might want:

- **Stratified sampling**: Ensures each class is represented proportionally in every batch
- **Weighted sampling**: Gives more weight to underrepresented or important samples
- **Curriculum learning**: Starts with easy examples and gradually increases difficulty

Example usage:

## How It Works

Data samplers control how samples are selected for each epoch of training.
Different sampling strategies can improve training convergence and handle
imbalanced datasets.

## Properties

| Property | Summary |
|:-----|:--------|
| `Length` | Gets the total number of samples this sampler will produce per epoch. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetIndices` | Returns an enumerable of indices for one epoch of sampling. |
| `OnEpochStart(Int32)` | Called at the start of each epoch to allow the sampler to adjust its behavior. |
| `SetSeed(Int32)` | Sets the random seed for reproducible sampling. |

