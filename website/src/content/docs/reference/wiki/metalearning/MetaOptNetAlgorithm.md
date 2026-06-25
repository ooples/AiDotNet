---
title: "MetaOptNetAlgorithm<T, TInput, TOutput>"
description: "Implementation of Meta-learning with Differentiable Convex Optimization (MetaOptNet) algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Meta-learning with Differentiable Convex Optimization (MetaOptNet) algorithm.

## For Beginners

Imagine you're trying to fit a line to some points.
MAML would iteratively adjust the line: "move a bit left, now a bit right..."
MetaOptNet uses math to find the exact best line in one shot using the formula:
w = (X^T X + λI)^(-1) X^T y

## How It Works

MetaOptNet replaces the gradient-based inner-loop optimization of MAML with a
differentiable convex optimization solver. This provides several advantages:

**Key Innovation:** Instead of gradient descent in the inner loop:

**Supported Solvers:**

- Ridge Regression: Fast, closed-form, good for most tasks
- SVM: More powerful, better margins, but slower
- Logistic Regression: For probabilistic outputs

**Algorithm:**

Reference: Lee, K., Maji, S., Ravichandran, A., & Soatto, S. (2019).
Meta-Learning with Differentiable Convex Optimization. CVPR 2019.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaOptNetAlgorithm(MetaOptNetOptions<,,>)` | Initializes a new instance of the MetaOptNetAlgorithm class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` | Gets the algorithm type identifier for this meta-learner. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts the meta-learned model to a new task using convex optimization. |
| `ApplySoftmax(Matrix<>)` | Applies softmax to logit matrix. |
| `ComputeEncoderGradients(IMetaLearningTask<,,>,Matrix<>,Matrix<>,Matrix<>,)` | Computes gradients for the encoder. |
| `ComputeLogisticGradient(Matrix<>,Matrix<>,Matrix<>)` | Computes logistic regression gradient. |
| `ComputeLogits(Matrix<>,Matrix<>)` | Computes logits using classifier weights. |
| `ComputeSoftmaxCrossEntropy(Vector<>,Matrix<>,Int32,Int32)` | Computes the per-sample averaged softmax cross-entropy between flattened logits [numSamples × numClasses] and one-hot labels [numSamples, numClasses]. |
| `ComputeTemperatureGradient(Vector<>,,)` | Computes gradient with respect to temperature using paper-faithful softmax CE (Lee et al. |
| `ConvertFromVector(Vector<>)` | Converts a vector to the output type. |
| `ConvertToLabels()` | Converts output to one-hot label matrix. |
| `ExtractEmbeddings(,Int32)` | Extracts embeddings from input, sized to `expectedSamples` rows. |
| `MatrixMultiply(Matrix<>,Matrix<>)` | Multiplies two matrices. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step using MetaOptNet's convex optimization approach. |
| `NormalizeEmbeddings(Matrix<>)` | Normalizes embeddings to unit norm. |
| `ScaleByTemperature(Vector<>,)` | Scales logits by temperature. |
| `SolveConvexProblem(Matrix<>,Matrix<>)` | Solves the convex optimization problem to get classifier weights. |
| `SolveLinearSystem(Matrix<>,Matrix<>)` | Solves linear system Ax = b using iterative refinement. |
| `SolveLogisticRegression(Matrix<>,Matrix<>)` | Solves logistic regression using Newton's method. |
| `SolveRidgeRegression(Matrix<>,Matrix<>)` | Solves ridge regression: w* = (X^T X + λI)^(-1) X^T y |
| `SolveSVM(Matrix<>,Matrix<>)` | Solves SVM using simplified quadratic programming. |

