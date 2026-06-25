---
title: "CompressionAwareFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that considers both model accuracy and compression effectiveness."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that considers both model accuracy and compression effectiveness.

## For Beginners

When compressing a model, you have competing goals:

1. **Keep accuracy high** - You don't want the compressed model to make worse predictions
2. **Make the model smaller** - Smaller models use less memory and storage
3. **Make inference faster** - Smaller models often run faster

This fitness calculator combines all three goals into a single score that optimization
algorithms (like AutoML or genetic algorithms) can use to find the best compression settings.

The weights control how much each goal matters:

- High accuracyWeight = prioritize keeping predictions accurate
- High compressionWeight = prioritize making the model smaller
- High speedWeight = prioritize making inference faster

Example:

- For edge devices with limited memory: increase compressionWeight
- For real-time applications: increase speedWeight
- For medical/financial applications: increase accuracyWeight

## How It Works

This fitness calculator implements multi-objective optimization for model compression.
It balances accuracy preservation against compression benefits (size reduction, speed improvement).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CompressionAwareFitnessCalculator(IFitnessCalculator<,,>,Double,Double,Double)` | Initializes a new instance of the CompressionAwareFitnessCalculator class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CompressionMetrics` | Gets or sets the current compression metrics to use in fitness calculation. |
| `IsHigherScoreBetter` | Gets a value indicating whether higher fitness scores are better. |
| `PreferredDataSetType` | Forwards the wrapped calculator's preferred dataset type when it implements `IPreferredDataSetFitnessCalculator`; otherwise falls back to `Validation`, which preserves the historical default for external calculators that don't opt in to the… |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateFitnessScore(DataSetStats<,,>)` | Calculates a fitness score from dataset statistics. |
| `CalculateFitnessScore(ModelEvaluationData<,,>)` | Calculates a composite fitness score considering both accuracy and compression. |
| `IsBetterFitness(,)` | Compares two fitness scores and determines if the current is better than the best. |
| `NormalizeBaseFitness()` | Normalizes the base fitness score to a [0, 1] range. |

