---
title: "IResamplingStrategy<T>"
description: "Defines the interface for resampling strategies used to handle imbalanced datasets."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the interface for resampling strategies used to handle imbalanced datasets.

## For Beginners

In many real-world problems, one class has far fewer examples than another.
For example:

- Fraud detection: 99% normal transactions, 1% fraudulent
- Disease diagnosis: 95% healthy, 5% diseased
- Spam filtering: 80% legitimate, 20% spam

This imbalance causes problems because machine learning models tend to ignore the minority class
and just predict the majority class for everything. Resampling strategies fix this by:

1. **Oversampling:** Creating more examples of the minority class (like SMOTE, ADASYN)
2. **Undersampling:** Removing examples from the majority class (like RandomUnderSampler)
3. **Combined:** Doing both (like SMOTEENN, SMOTETomek)

After resampling, the model sees a more balanced dataset and learns to recognize both classes.

## How It Works

Resampling strategies modify the training data to address class imbalance, either by
creating synthetic samples for minority classes (oversampling) or removing samples
from majority classes (undersampling).

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of the resampling strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetStatistics` | Gets statistics about the resampling operation. |
| `Resample(Matrix<>,Vector<>)` | Resamples the dataset to address class imbalance. |

