---
title: "VariationalContinualLearning<T>"
description: "Implements Variational Continual Learning (VCL) for Bayesian continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning`

Implements Variational Continual Learning (VCL) for Bayesian continual learning.

## For Beginners

VCL uses Bayesian neural networks where each weight has a
probability distribution (mean and variance) rather than a single value. This allows
the network to represent uncertainty and naturally prevents forgetting by using the
posterior from previous tasks as the prior for new tasks.

## How It Works

**How it works:**

**Key Formula:**

Loss = E_q[log p(D|w)] - KL(q(w|D_new) || p(w|D_old))

where q(w|D_new) is the current posterior and p(w|D_old) is the previous posterior (now prior).

**Advantages:**

**Reference:** Nguyen, C.V., Li, Y., Bui, T.D., and Turner, R.E.
"Variational Continual Learning" (2018). ICLR.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VariationalContinualLearning(Double,Double,Nullable<Int32>)` | Initializes a new instance of the VariationalContinualLearning class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccumulatesAcrossTasks` |  |
| `InitialLogVar` | Gets the initial log-variance value. |
| `TaskCount` | Gets the number of tasks processed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AfterTask(INeuralNetwork<>,ValueTuple<Tensor<>,Tensor<>>,Int32)` |  |
| `BeforeTask(INeuralNetwork<>,Int32)` |  |
| `ComputeLoss(INeuralNetwork<>)` |  |
| `Exp()` | Computes exp(x) for a value. |
| `GetPosterior` | Gets the current posterior distribution parameters. |
| `GetPrior` | Gets the current prior distribution parameters. |
| `Log()` | Computes log(x) for a value. |
| `ModifyGradients(INeuralNetwork<>,Vector<>)` |  |
| `Reset` |  |
| `SampleStandardNormal` | Samples from standard normal distribution. |
| `SampleWeights(INeuralNetwork<>)` | Samples weights from the posterior distribution for prediction. |
| `Sqrt()` | Computes sqrt(x) for a value. |
| `UpdateVariance(Vector<>,Double)` | Updates the posterior log-variance based on gradient information. |

