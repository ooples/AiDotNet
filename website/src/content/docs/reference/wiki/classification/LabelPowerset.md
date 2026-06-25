---
title: "LabelPowerset<T>"
description: "Implements the Label Powerset approach for multi-label classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.MultiLabel`

Implements the Label Powerset approach for multi-label classification.

## For Beginners

Label Powerset takes a clever approach:

Instead of predicting labels independently, it treats each unique combination of labels as a
single class. For example, if you have 3 labels (action, comedy, romance):

- "none" becomes class 0
- "action only" becomes class 1
- "comedy only" becomes class 2
- "romance only" becomes class 3
- "action+comedy" becomes class 4
- "action+romance" becomes class 5
- "comedy+romance" becomes class 6
- "action+comedy+romance" becomes class 7

Now we train ONE multi-class classifier that directly predicts which combination applies.

Pros:

- Perfectly captures label correlations (impossible combinations never predicted)
- Only one classifier to train
- Naturally handles label interdependencies

Cons:

- Number of possible classes = 2^n (exponential in number of labels)
- Many classes may have very few examples (data sparsity)
- New label combinations unseen in training cannot be predicted

Works best when:

- Number of labels is small (≤10)
- Label combinations in test data were seen in training
- Label correlations are important

## How It Works

Label Powerset transforms the multi-label problem into a single multi-class problem by treating
each unique combination of labels as a separate class.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LabelPowerset` | Initializes a new instance with default settings using Gaussian Naive Bayes as the base classifier. |
| `LabelPowerset(Func<IClassifier<>>,ClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the LabelPowerset class with a classifier factory. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumLabelCombinations` | Gets the number of unique label combinations found in training data. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients to update model parameters. |
| `BuildLabelCombinationMappings(Matrix<>)` | Builds mappings between label combinations and class indices. |
| `Clone` | Creates a deep copy of this classifier. |
| `ComputeGradients(Matrix<>,Matrix<>,ILossFunction<>)` | Computes gradients for gradient-based optimization. |
| `CreateNewInstance` | Creates a new instance of this classifier with default configuration. |
| `Deserialize(Byte[])` |  |
| `GetLabelArray(Matrix<>,Int32)` | Extracts a boolean array representing the label combination for a row. |
| `GetLabelKey(Matrix<>,Int32)` | Creates a string key representing a label combination. |
| `GetLabelsForClass(Int32)` | Gets the label combination for a given class index. |
| `GetParameters` | Gets all learnable parameters of the model as a single vector. |
| `PredictMultiLabelProbabilities(Matrix<>)` | Predicts probabilities for each label for each sample. |
| `Serialize` |  |
| `SetParameters(Vector<>)` | Sets the parameters of this model. |
| `TrainMultiLabelCore(Matrix<>,Matrix<>)` | Core implementation of multi-label training using Label Powerset. |
| `TransformToClassLabels(Matrix<>)` | Transforms multi-label matrix into single-label class vector. |
| `WithParameters(Vector<>)` | Creates a new instance of the model with the specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_classToLabels` | Maps class indices to label combinations. |
| `_classifier` | The trained multi-class classifier. |
| `_classifierFactory` | Factory function to create the multi-class classifier. |
| `_labelsToClass` | Maps label combinations to class indices. |
| `_numCombinations` | The number of unique label combinations (classes) found in training data. |

