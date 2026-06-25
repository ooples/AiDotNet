---
title: "LabelPropagation<T>"
description: "Implements the Label Propagation algorithm for semi-supervised classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.SemiSupervised`

Implements the Label Propagation algorithm for semi-supervised classification.

## For Beginners

Imagine you have a few people in a social network whose political views
you know (labeled), and many others whose views you don't know (unlabeled). Label Propagation
assumes that connected people (friends) tend to have similar views.

The algorithm builds a graph where people are connected based on similarity, then lets the
known labels "spread" through the connections. After many rounds of spreading, even people
far from the labeled ones get assigned labels based on how the information flowed through
the network.

The key insight is that similar data points should have similar labels, and this similarity
can be captured through a graph structure.

## How It Works

Label Propagation is a graph-based semi-supervised learning algorithm that propagates labels
from labeled samples to unlabeled samples through a similarity graph.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LabelPropagation` | Initializes a new instance of the LabelPropagation class with default settings. |
| `LabelPropagation(IKernelFunction<>,Int32,,Nullable<Int32>)` | Initializes a new instance of the LabelPropagation class with specified parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients to update model parameters. |
| `BuildAffinityMatrix(Matrix<>)` | Builds the affinity (similarity) matrix using the kernel function. |
| `Clone` | Creates a deep copy of this classifier. |
| `CombineData(Matrix<>,Matrix<>)` | Combines labeled and unlabeled data into a single feature matrix. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` | Computes gradients for gradient-based optimization. |
| `CreateDefaultKernel` | Creates the default kernel (RBF/Gaussian). |
| `CreateNewInstance` | Creates a new instance of this classifier with default configuration. |
| `Deserialize(Byte[])` |  |
| `ExtractPseudoLabels` | Extracts pseudo-labels from the converged label distributions. |
| `GetClassIndex()` | Gets the index of a class label in the ClassLabels array. |
| `GetParameters` | Gets all learnable parameters of the model as a single vector. |
| `InitializeLabelDistributions(Vector<>,Int32)` | Initializes the label distribution matrix. |
| `MultiplyMatrices(Matrix<>,Matrix<>)` | Multiplies two matrices. |
| `NormalizeAffinity(Matrix<>)` | Normalizes the affinity matrix to create transition probabilities. |
| `Predict(Matrix<>)` | Predicts class labels for new samples. |
| `PredictProbabilities(Matrix<>)` | Predicts class probabilities for each sample. |
| `PredictProbabilitiesSingle(Vector<>)` | Predicts class probabilities for a single sample. |
| `PredictSingle(Vector<>)` | Predicts the class label for a single sample. |
| `PropagateLabels(Matrix<>,Matrix<>)` | Propagates labels through the graph iteratively. |
| `Serialize` |  |
| `SetParameters(Vector<>)` | Sets the parameters of this model. |
| `TrainSemiSupervisedCore(Matrix<>,Vector<>,Matrix<>)` | Core implementation of semi-supervised training using label propagation. |
| `TrainSupervisedCore(Matrix<>,Vector<>)` | Core implementation of supervised training (using only labeled data). |
| `WithParameters(Vector<>)` | Creates a new instance of the model with the specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_affinityMatrix` | The affinity matrix representing pairwise similarities between all samples. |
| `_allFeatures` | The combined feature matrix (labeled + unlabeled) after training. |
| `_kernel` | The kernel function used to compute similarity between samples. |
| `_labelDistributions` | The label distribution matrix where each row is a sample and each column is a class. |
| `_maxIterations` | Maximum number of iterations for label propagation. |
| `_numLabeled` | Number of labeled samples stored during training. |
| `_random` | Random number generator for tie-breaking. |
| `_tolerance` | Convergence tolerance for stopping the propagation early. |

