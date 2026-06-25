---
title: "RidgeClassifier<T>"
description: "Ridge Classifier - converts regression to classification using regularized least squares."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Linear`

Ridge Classifier - converts regression to classification using regularized least squares.

## For Beginners

Ridge Classifier treats classification as a regression problem:

How it works:

1. Convert class labels to numbers (-1 and +1 for binary)
2. Fit a ridge regression to these numbers
3. For prediction, output whichever class the regression is closest to

Why use Ridge Classifier:

- Very fast training (closed-form solution)
- Works well when number of features is large
- Stable due to regularization
- Good baseline classifier

Trade-offs:

- Doesn't optimize classification accuracy directly
- May not work as well as logistic regression for probability estimates
- Assumes linear relationship between features and class labels

## How It Works

Ridge Classifier uses ridge regression (L2 regularized least squares) and then
converts the continuous predictions to class labels.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RidgeClassifier(LinearClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the RidgeClassifier class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CenterMatrix(Matrix<>,Vector<>)` | Centers a matrix by subtracting column means. |
| `CenterVector(Vector<>,)` | Centers a vector by subtracting the mean. |
| `Clone` |  |
| `ComputeMean(Matrix<>)` | Computes the mean of each column in a matrix. |
| `ComputeMean(Vector<>)` | Computes the mean of a vector. |
| `CreateNewInstance` |  |
| `SolveLinearSystem(Matrix<>,Vector<>)` | Solves Ax = b using Gaussian elimination with partial pivoting. |
| `Train(Matrix<>,Vector<>)` | Trains the Ridge Classifier using closed-form solution. |

