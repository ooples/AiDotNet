---
title: "RandomUnderSampler<T>"
description: "Implements random undersampling to balance imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Tabular.Undersampling`

Implements random undersampling to balance imbalanced datasets.

## For Beginners

Random undersampling reduces the majority class by randomly
removing samples until the classes are balanced. This is the simplest undersampling method.

## How It Works

**How it works:**

- Count samples in majority and minority classes
- Randomly select N samples from the majority class
- N is determined by the target ratio

**Advantages:**

- Simple and fast
- Reduces training time by having fewer samples
- May improve some classifiers by removing redundant samples

**Disadvantages:**

- May discard useful information from the majority class
- Can lead to underfitting if too many samples are removed
- Random selection doesn't consider sample importance

**When to use:**

- When the majority class is very large and redundant
- When training time is a concern
- As a baseline before trying more sophisticated methods

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomUnderSampler(Double,Boolean,Nullable<Int32>)` | Creates a new random undersampler. |

## Properties

| Property | Summary |
|:-----|:--------|
| `RandomSeed` | Gets the random seed for reproducibility. |
| `SamplingRatio` | Gets the target ratio between minority and majority samples after undersampling. |
| `WithReplacement` | Gets whether to use replacement when sampling. |

## Methods

| Method | Summary |
|:-----|:--------|
| `SelectRandomIndices(List<Int32>,Int32,Boolean,Random)` | Randomly selects indices from the given list. |
| `Undersample(Matrix<>,Vector<>,)` | Performs random undersampling on the majority class. |

