---
title: "ClassifierChainClassifier<T>"
description: "Implements the Classifier Chain approach for multi-label classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.MultiLabel`

Implements the Classifier Chain approach for multi-label classification.

## For Beginners

Classifier Chain improves on Binary Relevance by capturing label correlations:

Consider predicting movie genres:

1. First classifier asks: "Is this a sequel?" (uses only movie features)
2. Second classifier asks: "Is this action?" (uses movie features + "is sequel" prediction)
3. Third classifier asks: "Is this comedy?" (uses movie features + "is sequel" + "is action")
4. And so on...

By including previous predictions as features, each classifier can learn from label dependencies.
For example, if a movie is classified as "sequel", the next classifier knows this and can adjust
its predictions accordingly (sequels are more likely to be action movies).

Pros:

- Captures label dependencies
- Still relatively simple to implement
- Often outperforms Binary Relevance

Cons:

- Chain order matters (different orders can give different results)
- Errors can propagate through the chain
- Order selection can be tricky

The chain order can be specified manually or determined randomly. For best results, consider
training multiple chains with different orders and combining their predictions (Ensemble of Classifier Chains).

## How It Works

Classifier Chain is an extension of Binary Relevance that models label dependencies by training
classifiers in a chain, where each classifier uses the predictions of previous classifiers as
additional features.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClassifierChainClassifier` | Initializes a new instance with default settings using Gaussian Naive Bayes as the base classifier. |
| `ClassifierChainClassifier(Func<IClassifier<>>,Int32[],Boolean,Nullable<Int32>,ClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the ClassifierChainClassifier class with a classifier factory. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ChainOrder` | Gets the chain order used for training and prediction. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients to update model parameters. |
| `Clone` | Creates a deep copy of this classifier. |
| `ComputeGradients(Matrix<>,Matrix<>,ILossFunction<>)` | Computes gradients for gradient-based optimization. |
| `CreateAugmentedFeatures(Matrix<>,Matrix<>,Int32)` | Creates augmented features for a position in the chain. |
| `CreateAugmentedFeaturesForPrediction(Matrix<>,Matrix<>,Int32)` | Creates augmented features for prediction using previous predictions. |
| `CreateNewInstance` | Creates a new instance of this classifier with default configuration. |
| `Deserialize(Byte[])` |  |
| `DetermineChainOrder` | Determines the order of labels in the chain. |
| `GetParameters` | Gets all learnable parameters of the model as a single vector. |
| `PredictMultiLabelProbabilities(Matrix<>)` | Predicts probabilities for each label for each sample. |
| `Serialize` |  |
| `SetParameters(Vector<>)` | Sets the parameters of this model. |
| `TrainMultiLabelCore(Matrix<>,Matrix<>)` | Core implementation of multi-label training using Classifier Chain. |
| `WithParameters(Vector<>)` | Creates a new instance of the model with the specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_chainClassifiers` | The trained binary classifiers in chain order. |
| `_chainOrder` | The order of labels in the chain (indices into the label array). |
| `_classifierFactory` | Factory function to create binary classifiers for each position in the chain. |
| `_random` | Random number generator for shuffling chain order. |
| `_specifiedOrder` | User-specified chain order (null means use natural or random order). |
| `_useRandomOrder` | Whether to use a random chain order. |

