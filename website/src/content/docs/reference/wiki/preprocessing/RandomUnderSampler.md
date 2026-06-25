---
title: "RandomUnderSampler<T>"
description: "Implements random undersampling for handling imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.ImbalancedLearning`

Implements random undersampling for handling imbalanced datasets.

## For Beginners

If you have 1000 "normal" samples and 100 "fraud" samples,
random undersampling might randomly select 100-200 of the "normal" samples to keep,
discarding the rest.

Advantages:

- Very simple and fast
- Good baseline to compare against
- Works well with large datasets where losing data isn't critical

Disadvantages:

- Randomly discards potentially useful information
- May remove important samples near decision boundary
- Results can vary significantly based on random selection

When to use:

- Large datasets where you can afford to lose data
- As a quick baseline before trying more sophisticated methods
- When you need fast results and perfect balance

References:

- Kotsiantis et al. (2006). "Handling imbalanced datasets: A review"

## How It Works

Random undersampling randomly removes samples from the majority class until
the desired balance is achieved. It's the simplest undersampling method.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomUnderSampler(Double,Boolean,Nullable<Int32>)` | Initializes a new instance of the RandomUnderSampler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this undersampling strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `SelectSamplesToKeep(Matrix<>,Vector<>,List<Int32>,List<Int32>,Int32)` | Selects which majority samples to keep using random selection. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_replacement` | Whether to replace samples when selecting (sampling with replacement). |

