---
title: "InterventionalDistribution<T>"
description: "Represents the interventional distribution P(Y | do(X = x)) from Pearl's do-calculus."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery`

Represents the interventional distribution P(Y | do(X = x)) from Pearl's do-calculus.

## For Beginners

Imagine you discover that ice cream sales and drowning rates are
correlated. The observational distribution P(Drowning | IceCream = high) is high.
But the interventional distribution P(Drowning | do(IceCream = high)) — what happens
if we FORCE ice cream sales to be high — is NOT high, because the causal arrow is
actually Temperature → IceCream and Temperature → Drowning. Interventions break the
confounding by the temperature variable.

## How It Works

An interventional distribution answers the question: "What would happen to variable Y if
we actively SET variable X to value x?" This is fundamentally different from conditioning
(P(Y | X = x)) because it breaks all causal arrows INTO X, simulating an experiment.

**Truncated Factorization Formula:**
P(Y | do(X = x)) = Σ_pa(X) P(Y | X, pa(X)) * P(pa(X))
where pa(X) are the parents of X in the causal graph.

**Usage:**

Reference: Pearl (2009), "Causality: Models, Reasoning, and Inference", Cambridge University Press, Ch. 3.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InterventionalDistribution(Int32,String,,Int32,String,Double[],Double)` | Creates a new InterventionalDistribution from computed interventional samples. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageCausalEffect` | Gets the Average Causal Effect (ACE): E[Y \| do(X=x)] - E[Y_observational]. |
| `InterventionValue` | Gets the value the intervention variable is set to. |
| `InterventionVariableIndex` | Gets the index of the intervention variable. |
| `InterventionVariableName` | Gets the name of the intervention variable (the variable being set via do-operator). |
| `Max` | Gets the maximum value in the interventional samples. |
| `Mean` | Gets the mean of the interventional distribution. |
| `Median` | Gets the median of the interventional distribution. |
| `Min` | Gets the minimum value in the interventional samples. |
| `SampleCount` | Gets the number of interventional samples. |
| `Samples` | Gets the interventional samples of the target variable under the do-intervention. |
| `StandardDeviation` | Gets the standard deviation of the interventional distribution. |
| `TargetVariableIndex` | Gets the index of the target variable. |
| `TargetVariableName` | Gets the name of the target variable whose distribution is being computed. |
| `Variance` | Gets the variance of the interventional distribution. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ConfidenceInterval(Double)` | Computes a confidence interval for the interventional mean. |
| `EstimateDensity(Double)` | Computes an empirical estimate of the probability density at a given value using a Gaussian kernel. |
| `Quantile(Double)` | Computes the specified quantile of the interventional distribution. |
| `ToString` |  |

