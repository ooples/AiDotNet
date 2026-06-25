---
title: "LabelSpreading<T>"
description: "Implements the Label Spreading algorithm for semi-supervised classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.SemiSupervised`

Implements the Label Spreading algorithm for semi-supervised classification.

## For Beginners

Label Spreading improves on Label Propagation in two ways:

1. **Symmetric Normalization:** Instead of just normalizing rows, it normalizes both

rows and columns. This makes the algorithm more stable when you have clusters of
very different sizes. Think of it like making sure influence flows equally in both
directions between connected points.

2. **Alpha Parameter:** This controls how much the original labels are preserved.

With alpha = 0.2, each iteration keeps 20% of the original label and takes 80% from
neighbors. This prevents the algorithm from "forgetting" the original labels while
still allowing information to spread.

The result is often more robust predictions, especially on noisy data or when clusters
have unequal sizes.

## How It Works

Label Spreading is a variant of Label Propagation that uses symmetric normalization
and a clamping factor (alpha) to balance between original labels and propagated information.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LabelSpreading` | Initializes a new instance of the LabelSpreading class with default settings. |
| `LabelSpreading(IKernelFunction<>,Int32,,,Nullable<Int32>)` | Initializes a new instance of the LabelSpreading class with specified parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients to update model parameters. |
| `BuildAffinityMatrix(Matrix<>)` | Builds the affinity (similarity) matrix using the kernel function. |
| `Clone` | Creates a deep copy of this classifier. |
| `CloneMatrix(Matrix<>)` | Creates a deep copy of a matrix. |
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
| `NormalizeRows(Matrix<>)` | Normalizes each row of a matrix to sum to 1. |
| `Predict(Matrix<>)` | Predicts class labels for new samples. |
| `PredictProbabilities(Matrix<>)` | Predicts class probabilities for each sample. |
| `PredictProbabilitiesSingle(Vector<>)` | Predicts class probabilities for a single sample. |
| `PredictSingle(Vector<>)` | Predicts the class label for a single sample. |
| `Serialize` |  |
| `SetParameters(Vector<>)` | Sets the parameters of this model. |
| `SpreadLabels` | Spreads labels through the graph iteratively with clamping. |
| `SymmetricNormalize(Matrix<>)` | Applies symmetric normalization to the affinity matrix. |
| `TrainSemiSupervisedCore(Matrix<>,Vector<>,Matrix<>)` | Core implementation of semi-supervised training using label spreading. |
| `TrainSupervisedCore(Matrix<>,Vector<>)` | Core implementation of supervised training (using only labeled data). |
| `WithParameters(Vector<>)` | Creates a new instance of the model with the specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_allFeatures` | The combined feature matrix (labeled + unlabeled) after training. |
| `_alpha` | The clamping factor (alpha) that balances original labels vs. |
| `_initialDistributions` | The initial label distributions (before propagation). |
| `_kernel` | The kernel function used to compute similarity between samples. |
| `_labelDistributions` | The label distribution matrix where each row is a sample and each column is a class. |
| `_maxIterations` | Maximum number of iterations for label spreading. |
| `_normalizedAffinity` | The symmetrically normalized affinity matrix (Laplacian-style normalization). |
| `_numLabeled` | Number of labeled samples stored during training. |
| `_random` | Random number generator for tie-breaking. |
| `_tolerance` | Convergence tolerance for stopping the spreading early. |

