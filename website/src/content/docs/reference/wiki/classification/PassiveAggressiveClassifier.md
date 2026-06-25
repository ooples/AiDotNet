---
title: "PassiveAggressiveClassifier<T>"
description: "Passive-Aggressive classifier for online learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Linear`

Passive-Aggressive classifier for online learning.

## For Beginners

Unlike regular gradient descent, Passive-Aggressive:

1. Only updates when it makes a mistake
2. When it updates, it does the minimum needed to fix the mistake

It's great for:

- Online learning (data arrives one sample at a time)
- Streaming data
- When you want a balance between learning and stability

The regularization parameter C controls the aggressiveness:

- Higher C: More aggressive updates (may overfit to noise)
- Lower C: More passive (may underfit)

## How It Works

The Passive-Aggressive algorithm is an online learning algorithm that:

- Is "passive" when the prediction is correct (no update)
- Is "aggressive" when wrong (makes the minimal update to correct the mistake)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PassiveAggressiveClassifier(PassiveAggressiveOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the PassiveAggressiveClassifier class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Options` | Gets the PA classifier specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `ComputeTau(,,)` | Computes the update step tau based on the PA variant. |
| `CreateNewInstance` |  |
| `GetModelMetadata` |  |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |

