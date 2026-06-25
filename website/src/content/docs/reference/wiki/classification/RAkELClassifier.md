---
title: "RAkELClassifier<T>"
description: "Implements RAkEL (Random k-Labelsets) for multi-label classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.MultiLabel`

Implements RAkEL (Random k-Labelsets) for multi-label classification.

## For Beginners

RAkEL solves multi-label classification by training multiple
Label Powerset classifiers on random subsets of k labels each. This captures label correlations
(like Label Powerset) while avoiding the exponential explosion of label combinations.

## How It Works

**How it works:**

- Randomly partition labels into overlapping subsets of size k
- Train a Label Powerset classifier on each subset
- Combine predictions by voting across all classifiers that predict each label

**Key parameters:**

- **k:** Size of each labelset (default: 3). Larger k captures more correlations but has more classes.
- **numLabelsets:** Number of random labelsets to create (default: 2*numLabels)

**Reference:** Tsoumakas et al., "Random k-Labelsets for Multilabel Classification" (2011)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RAkELClassifier(Int32,Int32,Nullable<Int32>,ClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Creates a new RAkEL classifier. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LabelsetSize` | Gets the size of each labelset (k parameter). |
| `NumLabelsets` | Gets the number of labelsets to create. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `GenerateRandomLabelset(Int32)` | Generates a random labelset of the specified size. |
| `GetParameters` |  |
| `PredictMultiLabelProbabilities(Matrix<>)` | Predicts multi-label probabilities using RAkEL voting. |
| `Serialize` |  |
| `SetParameters(Vector<>)` |  |
| `TrainMultiLabelCore(Matrix<>,Matrix<>)` | Core training implementation for RAkEL. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_inverseLabelMaps` | Inverse mapping from class index back to label combination. |
| `_labelCombinationMaps` | The unique label combination codes for each labelset classifier. |
| `_labelsetWeights` | Weight matrices for each labelset classifier. |
| `_labelsets` | The random labelsets (each is an array of label indices). |
| `_random` | Random instance for reproducibility. |

