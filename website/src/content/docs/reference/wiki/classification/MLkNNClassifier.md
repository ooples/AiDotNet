---
title: "MLkNNClassifier<T>"
description: "Implements ML-kNN (Multi-Label k-Nearest Neighbors) for multi-label classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.MultiLabel`

Implements ML-kNN (Multi-Label k-Nearest Neighbors) for multi-label classification.

## For Beginners

ML-kNN extends k-NN to multi-label problems using Bayesian inference.
For each label, it estimates the probability that a sample has the label given how many of its
k nearest neighbors have that label.

## How It Works

**How it works:**

- Find k nearest neighbors of the query sample
- Count how many neighbors have each label
- Use Bayesian inference with prior probabilities from training data
- Predict label if P(label=1|neighbors) > P(label=0|neighbors)

**Key formula:**
P(H_l | E_l) = P(E_l | H_l) * P(H_l) / P(E_l)
where H_l = label l is present, E_l = count of neighbors with label l

**Reference:** Zhang & Zhou, "ML-KNN: A lazy learning approach to multi-label learning" (2007)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MLkNNClassifier(MLkNNOptions<>)` | Creates a new ML-kNN classifier. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `PredictMultiLabelProbabilities(Matrix<>)` |  |
| `Serialize` |  |
| `SetParameters(Vector<>)` |  |
| `TrainMultiLabelCore(Matrix<>,Matrix<>)` |  |

