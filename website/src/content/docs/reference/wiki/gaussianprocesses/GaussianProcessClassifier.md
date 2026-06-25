---
title: "GaussianProcessClassifier<T>"
description: "Implements a Gaussian Process Classifier using Laplace approximation for probabilistic classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Implements a Gaussian Process Classifier using Laplace approximation for probabilistic classification.

## For Beginners

A Gaussian Process Classifier (GPC) is a powerful machine learning method
that not only classifies data but also tells you how confident it is about each prediction.

Imagine you're building a spam filter:

- A regular classifier might say: "This email is spam"
- A GP classifier says: "This email is 95% likely to be spam, and I'm quite confident about this"

How does it work?

1. It learns a "latent function" - a hidden score for each point in your data space
2. This score is passed through a sigmoid function to get a probability
3. The Laplace approximation helps us handle the mathematical complexity

The Laplace Approximation:

- GP classification doesn't have a nice closed-form solution like GP regression
- The Laplace approximation finds the most likely values (the "mode") of the latent function
- It then approximates the posterior as a Gaussian centered at this mode
- This gives us uncertainty estimates even though the true posterior is non-Gaussian

When to use GP Classification:

- When you need probability estimates, not just class labels
- When you have small to medium-sized datasets (up to a few thousand points)
- When uncertainty quantification is important (medical diagnosis, risk assessment)
- When your decision boundary might be non-linear

Limitations:

- Scales cubically O(n³) with dataset size due to matrix operations
- For larger datasets, consider Sparse GP Classification (SVGP)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaussianProcessClassifier(IKernelFunction<>,MatrixDecompositionType,Int32,Double)` | Initializes a new instance of the GaussianProcessClassifier class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumClasses` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddJitter(Matrix<>)` | Adds a small jitter term to the diagonal for numerical stability. |
| `CalculateBMatrix(Matrix<>)` | Calculates the B matrix used in the Newton-Raphson update. |
| `CalculateGradient(Vector<>)` | Calculates the gradient of the log-likelihood with respect to f. |
| `CalculateKernelMatrix(Matrix<>,Matrix<>)` | Calculates the kernel matrix between two sets of data points. |
| `CalculateKernelVector(Matrix<>,Vector<>)` | Calculates the kernel values between a set of data points and a single point. |
| `CalculateLogMarginalLikelihood` | Calculates the log marginal likelihood of the model. |
| `CalculateSigmoid(Vector<>)` | Calculates the logistic sigmoid function for a vector of values. |
| `CalculateSqrtW(Matrix<>)` | Calculates the square root of the diagonal W matrix. |
| `CalculateWMatrix(Vector<>)` | Calculates the W matrix (diagonal of the Hessian of negative log-likelihood). |
| `Fit(Matrix<>,Vector<>)` | Trains the Gaussian Process classifier on the provided data using Laplace approximation. |
| `GetLogMarginalLikelihood` |  |
| `OptimizeLatentFunction` | Performs Newton-Raphson optimization to find the mode of the posterior distribution. |
| `Predict(Vector<>)` |  |
| `PredictProbabilities(Matrix<>)` |  |
| `TransformLabels(Vector<>)` | Transforms class labels from {0, 1} to {-1, +1} format. |
| `UpdateKernel(IKernelFunction<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_K` | The kernel matrix calculated from the training data. |
| `_W` | The Hessian of the negative log-likelihood, used in the Laplace approximation. |
| `_X` | The matrix of input features from the training data. |
| `_decompositionType` | The method used to decompose matrices for solving linear systems. |
| `_f` | The latent function values at training points (the mode of the posterior). |
| `_kernel` | The kernel function that determines how similarity between data points is calculated. |
| `_logMarginalLikelihood` | The cached log marginal likelihood from the last fit. |
| `_maxIterations` | Maximum number of iterations for the Newton-Raphson optimization. |
| `_numClasses` | The number of classes detected during training. |
| `_numOps` | Operations for performing numeric calculations with the generic type T. |
| `_originalLabels` | The original class labels from training (before transformation). |
| `_tolerance` | Convergence tolerance for the Newton-Raphson optimization. |
| `_y` | The vector of class labels from the training data, transformed to +1/-1 for binary classification. |

