---
title: "FunctionalBase<T>"
description: "Base class for functional/ICA-based causal discovery algorithms (LiNGAM, ANM, etc.)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.CausalDiscovery.Functional`

Base class for functional/ICA-based causal discovery algorithms (LiNGAM, ANM, etc.).

## For Beginners

These methods use clever mathematical properties to figure out
which variable causes which. For example, if X causes Y with some noise added, the
noise pattern looks different depending on which direction you assume the causation
goes. These methods detect that asymmetry.

## How It Works

Functional methods exploit properties of the data-generating process to determine
causal direction. Unlike constraint-based methods (which only learn equivalence classes),
functional methods can often identify the unique causal DAG.

**Key assumption families:**

- **LiNGAM:** Linear model with non-Gaussian noise → uses ICA to identify structure
- **ANM:** Additive noise model Y = f(X) + N → exploits asymmetry in residuals
- **PNL:** Post-nonlinear model Y = g(f(X) + N) → generalizes ANM

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeColumnVariance(Matrix<>,Int32)` | Computes column variance from a Matrix using generic operations. |
| `ComputeCorrelation(Vector<>,Vector<>)` | Computes Pearson correlation between two vectors. |
| `ComputeKurtosis(Vector<>)` | Computes kurtosis (fourth standardized moment) of a data vector. |
| `GaussianMI(Vector<>,Vector<>)` | Computes mutual information (Gaussian approximation) between two vectors. |
| `KernelRegressOut(Vector<>,Vector<>)` | Kernel regression residuals: fits response = f(predictor) + ε using Nadaraya–Watson, returns the residual vector ε. |
| `KernelSmooth(Matrix<>,Int32,Vector<>)` | Nadaraya–Watson kernel smoother for a single predictor column against a response vector. |
| `RegressOut(Vector<>,Vector<>)` | Computes residuals of y after regressing on x using generic Vector operations. |
| `StandardizeData(Matrix<>)` | Standardizes data to zero mean and unit variance using generic Matrix operations. |

