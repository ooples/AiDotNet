---
title: "SemiSupervisedClassifierBase<T>"
description: "Provides a base implementation for semi-supervised classification algorithms."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Classification.SemiSupervised`

Provides a base implementation for semi-supervised classification algorithms.

## For Beginners

This base class provides the foundation for building classifiers
that can learn from both labeled data (where you know the answers) and unlabeled data
(where you don't).

Think of it like a student learning with two types of study materials:

- A teacher's answer key (labeled data) - few examples but definitely correct
- Practice problems without answers (unlabeled data) - many examples to learn patterns from

By combining both, the student (classifier) can learn more effectively than using
just the answer key alone.

## How It Works

This abstract class extends ClassifierBase with semi-supervised learning capabilities,
allowing derived classes to leverage both labeled and unlabeled data during training.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SemiSupervisedClassifierBase(ClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>,ILossFunction<>)` | Initializes a new instance of the SemiSupervisedClassifierBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumLabeledSamples` | Gets or sets the number of labeled samples used in training. |
| `NumUnlabeledSamples` | Gets or sets the number of unlabeled samples used in training. |
| `PseudoLabelConfidences` | Stores the confidence scores for the pseudo-labels. |
| `PseudoLabels` | Stores the pseudo-labels assigned to unlabeled data during training. |
| `UnlabeledData` | The unlabeled feature matrix stored for prediction and analysis. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetPseudoLabelConfidences` | Gets the confidence scores for the pseudo-labels. |
| `GetPseudoLabels` | Gets the pseudo-labels assigned to the unlabeled data during training. |
| `Train(Matrix<>,Vector<>)` | Trains the classifier using only labeled data (standard supervised training). |
| `TrainSemiSupervised(Matrix<>,Vector<>,Matrix<>)` | Trains the classifier using both labeled and unlabeled data. |
| `TrainSemiSupervisedCore(Matrix<>,Vector<>,Matrix<>)` | Core implementation of semi-supervised training to be implemented by derived classes. |
| `TrainSupervisedCore(Matrix<>,Vector<>)` | Core implementation of standard supervised training. |
| `ValidateSemiSupervisedInput(Matrix<>,Vector<>,Matrix<>)` | Validates the input data for semi-supervised training. |

