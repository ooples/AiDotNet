---
title: "ConvexSolverType"
description: "Types of convex solvers available for MetaOptNet."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Types of convex solvers available for MetaOptNet.

## For Beginners

Instead of iteratively updating weights, convex solvers find the optimal solution directly.
Think of it like finding the lowest point in a bowl - convex problems have a single
lowest point, so we can find it mathematically without searching around.

## How It Works

MetaOptNet uses convex optimization in the inner loop instead of gradient descent.
These solver types provide different trade-offs between speed and classification power.

## Fields

| Field | Summary |
|:-----|:--------|
| `LogisticRegression` | Logistic regression solved via Newton's method. |
| `RidgeRegression` | Ridge regression (L2-regularized least squares). |
| `SVM` | Support Vector Machine with quadratic programming. |

