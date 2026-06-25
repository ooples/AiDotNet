---
title: "SelfTrainingClassifier<T>"
description: "A self-training classifier that iteratively labels high-confidence unlabeled samples."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.SemiSupervised`

A self-training classifier that iteratively labels high-confidence unlabeled samples.

## For Beginners

Self-training is like a student who:

1. Studies the examples with answers (labeled data)
2. Makes predictions on practice problems (unlabeled data)
3. Is most confident about some answers
4. Treats those confident answers as correct and studies them too
5. Repeats until no more confident predictions can be made

The key insight is that the classifier's most confident predictions are likely correct,
so we can use them to expand our training set. Over time, the classifier becomes
more accurate as it learns from its own confident predictions.

Algorithm steps:

1. Train base classifier on labeled data
2. Predict probabilities for all unlabeled samples
3. Find predictions with confidence above threshold
4. Add those samples (with predicted labels) to the training set
5. Repeat until no new samples are added or max iterations reached

References:

- Yarowsky, D. (1995). "Unsupervised word sense disambiguation rivaling supervised methods"
- Triguero et al. (2015). "Self-labeled techniques for semi-supervised learning"

## How It Works

Self-training is one of the oldest and simplest semi-supervised learning algorithms.
It works by iteratively training a classifier on labeled data, using it to predict
labels for unlabeled data, and adding high-confidence predictions to the training set.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelfTrainingClassifier` | Initializes a new instance with default settings using Gaussian Naive Bayes as the base classifier. |
| `SelfTrainingClassifier(IClassifier<>,Double,Int32,Int32,SelfTrainingClassifier<>.SelectionCriterion,ClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the SelfTrainingClassifier class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IterationsPerformed` | Gets the number of iterations performed during training. |
| `SamplesAdded` | Gets the number of samples added from unlabeled data during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients to update the model parameters. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` | Computes gradients for the model parameters. |
| `CreateNewInstance` | Creates a new instance of this classifier. |
| `Deserialize(Byte[])` | Deserializes the self-training classifier including its wrapped base classifier. |
| `GetConfidenceScores(Matrix<>,Vector<>)` | Gets confidence scores for predictions. |
| `GetParameters` | Gets all model parameters as a single vector. |
| `ListToMatrix(List<Vector<>>)` | Converts a list of vectors to a matrix. |
| `Predict(Matrix<>)` | Predicts class labels for the given input data. |
| `SelectSamplesToAdd(Matrix<>,List<Int32>)` | Selects unlabeled samples to add to the training set based on the selection criterion. |
| `Serialize` | Serializes the self-training classifier including its wrapped base classifier. |
| `SetParameters(Vector<>)` | Sets the parameters for this model. |
| `TrainSemiSupervisedCore(Matrix<>,Vector<>,Matrix<>)` | Core implementation of semi-supervised self-training. |
| `TrainSupervisedCore(Matrix<>,Vector<>)` | Core implementation of standard supervised training. |
| `WithParameters(Vector<>)` | Creates a new instance of the model with specified parameters. |

