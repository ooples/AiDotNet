---
title: "KraskovMIAlgorithm<T>"
description: "Kraskov MI — Mutual Information estimation using k-nearest neighbors (KSG estimator)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.InformationTheoretic`

Kraskov MI — Mutual Information estimation using k-nearest neighbors (KSG estimator).

## For Beginners

Most MI estimators assume the data follows a specific distribution
(like Gaussian). The Kraskov method doesn't make this assumption — it works by looking
at how close data points are to each other in different ways. This makes it more reliable
for complex, real-world data.

## How It Works

The Kraskov-Stoegbauer-Grassberger (KSG) estimator computes mutual information
using nearest-neighbor distances in the joint and marginal spaces. It's non-parametric
and works well for both linear and nonlinear dependencies.

**Algorithm (KSG Algorithm 1):**

- For each point, find its k-th nearest neighbor in the joint (X,Y) space using Chebyshev distance
- Let epsilon_i = distance to k-th neighbor in joint space
- Count n_x(i) = number of points with |x_j - x_i| < epsilon_i
- Count n_y(i) = number of points with |y_j - y_i| < epsilon_i
- MI = psi(k) - <psi(n_x + 1) + psi(n_y + 1)> + psi(N)

Reference: Kraskov et al. (2004), "Estimating Mutual Information", Physical Review E.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

