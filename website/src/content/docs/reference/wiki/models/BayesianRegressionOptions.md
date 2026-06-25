---
title: "BayesianRegressionOptions<T>"
description: "Configuration options for Bayesian regression algorithms."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Bayesian regression algorithms.

## For Beginners

Bayesian regression is like traditional regression (finding relationships between 
variables) but with an added layer of confidence information. Instead of just saying "we think y = 2x + 3," 
Bayesian regression says "we think y = 2x + 3, and we're 90% confident that the 2 is between 1.8 and 2.2." 
This approach is especially useful when you have limited data or want to incorporate prior knowledge about 
what the relationship might be. It helps you understand not just what the relationship is, but also how certain 
you can be about that relationship.

## How It Works

Bayesian regression is a statistical approach that applies Bayes' theorem to regression analysis.
Unlike traditional regression which produces point estimates, Bayesian regression provides probability
distributions for the model parameters, allowing for uncertainty quantification in predictions.

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets or sets the alpha parameter, which controls the precision of the prior distribution. |
| `Beta` | Gets or sets the beta parameter, which controls the precision of the likelihood function. |
| `Coef0` | Gets or sets the independent term (coef0) in Polynomial and Sigmoid kernels. |
| `DecompositionType` | Gets or sets the matrix decomposition method used for solving linear systems. |
| `Gamma` | Gets or sets the gamma parameter used in RBF, Polynomial, and Sigmoid kernels. |
| `KernelType` | Gets or sets the type of kernel function to use in the regression model. |
| `LaplacianGamma` | Gets or sets the gamma parameter for the Laplacian kernel. |
| `PolynomialDegree` | Gets or sets the degree of the Polynomial kernel. |

