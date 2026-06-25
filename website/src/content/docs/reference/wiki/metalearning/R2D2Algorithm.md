---
title: "R2D2Algorithm<T, TInput, TOutput>"
description: "Implementation of R2-D2 (Meta-learning with Differentiable Closed-form Solvers) for few-shot learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of R2-D2 (Meta-learning with Differentiable Closed-form Solvers) for few-shot learning.

## For Beginners

R2-D2 is a clever approach to few-shot learning:

**The Key Insight:**
Instead of slowly learning a classifier through gradient descent (like MAML does),
R2-D2 computes the optimal classifier instantly using a mathematical formula.

**How it works:**

1. Pass support set through the feature extractor to get features
2. Solve ridge regression on those features: w = (X^T X + lambda I)^-1 X^T y
3. This gives the OPTIMAL linear classifier in one step!
4. Evaluate on query set using this optimal classifier
5. Backpropagate through the entire process (including the matrix solve)
6. Update the feature extractor to produce better features

**Why it works:**

- The ridge regression formula has a known, exact derivative
- We can backpropagate through matrix inversion (using implicit differentiation)
- This trains the feature extractor to produce features that are easy to classify

**Analogy:**
Traditional few-shot (MAML):
"Here are 5 photos. Let me practice classifying them for 10 rounds... okay, now I'm ready."
R2-D2:
"Here are 5 photos. *does instant math* I know the optimal classifier. Done."

## How It Works

R2-D2 replaces MAML's iterative inner-loop gradient descent with a differentiable closed-form
ridge regression solver. The feature extractor (backbone) is meta-learned, and the final
classifier is computed analytically using ridge regression on the extracted features.

**Algorithm - R2-D2:**

**Key Insights:**

1. **Closed-Form = No Inner Loop**: Ridge regression has an exact solution, so there's

no iterative optimization. This is both faster and more stable than MAML.

2. **Differentiable Matrix Solve**: The gradient of the loss with respect to the features

flows through the matrix inversion using the identity: d(A^-1)/dA = -A^-1 (dA) A^-1.

3. **Feature Learning**: The backbone learns to produce features where ridge regression

works well, which means linearly separable features with good margins.

4. **Woodbury Identity**: When n_support << d (few-shot), we can use the Woodbury

identity to invert a smaller n x n matrix instead of d x d.

Reference: Bertinetto, L., Henriques, J. F., Torr, P., & Vedaldi, A. (2019).
Meta-learning with Differentiable Closed-form Solvers. ICLR 2019.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `R2D2Algorithm(R2D2Options<,,>)` | Initializes a new instance of the R2-D2 algorithm. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |
| `Lambda` | Gets the current lambda (regularization) value. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts to a new task by computing the ridge regression classifier on support features. |
| `ComputeRidgeWeights(Vector<>,Vector<>)` | Computes ridge regression weights from support features and labels using proper matrix ridge regression: w = (X^T X + lambda I)^-1 X^T y. |
| `EstimateFeatureDim(Int32,Int32)` | Estimates the per-example feature dimensionality from flattened vector lengths. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step using differentiable ridge regression. |
| `SolveRidgeSystem(List<Vector<>>,Vector<>,Int32)` | Solves ridge regression and returns predictions for query features. |
| `SplitIntoVectors(Vector<>,Int32,Int32)` | Splits a flat feature vector into a list of per-example multi-dimensional vectors. |
| `UpdateLambda(TaskBatch<,,>)` | Updates the lambda parameter using finite differences if meta-learning lambda. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lambda` | The current ridge regression regularization parameter (lambda). |

