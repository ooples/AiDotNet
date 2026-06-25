---
title: "BinaryRelevance<T>"
description: "Implements the Binary Relevance approach for multi-label classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.MultiLabel`

Implements the Binary Relevance approach for multi-label classification.

## For Beginners

Binary Relevance takes the "divide and conquer" approach:

Instead of trying to predict all labels at once, it trains a separate binary classifier for
each possible label. For example, if you're classifying movies into 5 genres:

1. Train a classifier that asks: "Is this movie action?" (yes/no)
2. Train a classifier that asks: "Is this movie comedy?" (yes/no)
3. Train a classifier that asks: "Is this movie drama?" (yes/no)
4. And so on for each genre...

To predict labels for a new movie, we run all 5 classifiers and combine their answers.

Pros:

- Simple to understand and implement
- Can use any binary classifier
- Parallelizable (each label classifier can train independently)

Cons:

- Ignores correlations between labels (e.g., "horror" and "thriller" often appear together)
- May produce inconsistent predictions (e.g., predicting "sequel" without "action")

For problems where labels are correlated, consider using Classifier Chains or Label Powerset instead.

## How It Works

Binary Relevance is the simplest multi-label classification method. It transforms the
multi-label problem into multiple independent binary classification problems, one for each label.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BinaryRelevance` | Initializes a new instance with default settings using Gaussian Naive Bayes as the base classifier. |
| `BinaryRelevance(Func<IClassifier<>>,ClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the BinaryRelevance class with a classifier factory. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients to update model parameters. |
| `Clone` | Creates a deep copy of this classifier. |
| `ComputeGradients(Matrix<>,Matrix<>,ILossFunction<>)` | Computes gradients for gradient-based optimization. |
| `CreateNewInstance` | Creates a new instance of this classifier with default configuration. |
| `Deserialize(Byte[])` |  |
| `GetParameters` | Gets all learnable parameters of the model as a single vector. |
| `PredictMultiLabelProbabilities(Matrix<>)` | Predicts probabilities for each label for each sample. |
| `Serialize` |  |
| `SetParameters(Vector<>)` | Sets the parameters of this model. |
| `TrainMultiLabelCore(Matrix<>,Matrix<>)` | Core implementation of multi-label training using Binary Relevance. |
| `WithParameters(Vector<>)` | Creates a new instance of the model with the specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_classifierFactory` | Factory function to create binary classifiers for each label. |
| `_labelClassifiers` | The trained binary classifiers, one per label. |

