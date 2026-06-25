---
title: "AIMGenerator<T>"
description: "AIM (Adaptive Iterative Mechanism) generator for differentially private synthetic data generation using marginal-based measurements and iterative optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

AIM (Adaptive Iterative Mechanism) generator for differentially private synthetic data
generation using marginal-based measurements and iterative optimization.

## For Beginners

AIM works like a privacy-preserving census:

1. First, simplify the data into categories (binning)
2. Then, privately count how many people are in each category combination
3. Finally, create synthetic data that matches those noisy counts

It's simpler than deep learning approaches but often works better for:

- Small datasets
- When you need formal privacy guarantees
- When training time is limited

## How It Works

AIM is a non-neural, statistical approach that works by:

No neural networks are used — only statistics and privacy-preserving mechanisms.

Reference: "AIM: An Adaptive and Iterative Mechanism for Differentially Private
Synthetic Data" (McKenna et al., 2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AIMGenerator(AIMOptions<>)` | Initializes a new AIM generator. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitInternal(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` |  |
| `GenerateInternal(Int32,Vector<>,Vector<>)` |  |

