---
title: "ISyntheticTabularGenerator<T>"
description: "Defines the contract for synthetic tabular data generators that learn the distribution of real tabular data and can produce new, realistic synthetic rows."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for synthetic tabular data generators that learn the distribution
of real tabular data and can produce new, realistic synthetic rows.

## For Beginners

Think of this like a "data factory" that learns from real data:

1. **Fit**: You show the generator your real table (like a spreadsheet).

It learns the patterns - ranges of numbers, common categories, relationships between columns.

2. **Generate**: The generator creates brand-new rows that look realistic

but aren't copies of any real data.

Common use cases:

- **Data augmentation**: Make a small dataset larger for better ML training
- **Privacy**: Share synthetic data instead of real data with sensitive information
- **Testing**: Generate realistic test data for development
- **Imbalanced data**: Generate more examples of rare categories

Example workflow:

## How It Works

Synthetic tabular data generators fit a model to real tabular data (containing a mix of
continuous/numerical and categorical/discrete columns), then generate new rows that
preserve the statistical properties and inter-column relationships of the original data.

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` | Gets the column metadata describing the structure of the data this generator was fitted on. |
| `IsFitted` | Gets a value indicating whether the generator has been fitted to data and is ready to generate. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` | Fits the generator model to the provided real tabular data. |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` | Asynchronously fits the generator model to the provided real tabular data. |
| `Generate(Int32,Vector<>,Vector<>)` | Generates new synthetic tabular data rows. |

