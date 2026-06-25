---
title: "PerceptronClassifier<T>"
description: "Classic Perceptron classifier - the original neural network building block."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Linear`

Classic Perceptron classifier - the original neural network building block.

## For Beginners

The Perceptron is the simplest possible neural network:

How it works:

1. Start with zero weights
2. For each training sample:
- If correct: do nothing
- If wrong: adjust weights in the direction of the correct class
3. Repeat until no mistakes (or max iterations)

Properties:

- Only works for linearly separable data
- Guaranteed to converge if data IS linearly separable
- Never converges if data is NOT linearly separable
- No notion of margin (unlike SVM)

Historical note: The Perceptron was invented in 1958 by Frank Rosenblatt
and was one of the first machine learning algorithms ever created!

## How It Works

The Perceptron is a linear classifier that updates weights only on mistakes.
It's historically significant as the foundation of neural networks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PerceptronClassifier(LinearClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the PerceptronClassifier class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateNewInstance` |  |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |

