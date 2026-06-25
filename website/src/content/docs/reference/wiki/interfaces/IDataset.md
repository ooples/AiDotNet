---
title: "IDataset<T, TInput, TOutput>"
description: "Interface for datasets used in active learning scenarios."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for datasets used in active learning scenarios.

## For Beginners

A dataset in machine learning is a collection of samples,
where each sample has input features (X) and optionally output labels (Y). This interface
provides a unified way to work with datasets in active learning.

## How It Works

**Key Concepts:**

**Active Learning Usage:**

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the number of samples in the dataset. |
| `HasLabels` | Gets whether this dataset has labels for all samples. |
| `Inputs` | Gets the input features for all samples. |
| `Outputs` | Gets the output labels for all samples. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddSamples([],[])` | Adds samples with labels to the dataset. |
| `Clone` | Creates a shallow copy of this dataset. |
| `Except(Int32[])` | Creates a subset of the dataset excluding the specified indices. |
| `GetIndices` | Gets the indices of all samples in this dataset. |
| `GetInput(Int32)` | Gets the input features for a specific sample. |
| `GetOutput(Int32)` | Gets the output label for a specific sample. |
| `GetSample(Int32)` | Gets both input and output for a specific sample. |
| `Merge(IDataset<,,>)` | Merges another dataset into this one. |
| `RemoveSamples(Int32[])` | Removes samples at the specified indices from the dataset. |
| `Shuffle(Random)` | Shuffles the dataset and returns a new shuffled dataset. |
| `Split(Double,Random)` | Splits the dataset into training and test sets. |
| `Subset(Int32[])` | Creates a subset of the dataset containing only the specified indices. |
| `UpdateLabels(Int32[],[])` | Updates the labels for specific samples. |

