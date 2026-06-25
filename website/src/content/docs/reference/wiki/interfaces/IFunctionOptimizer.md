---
title: "IFunctionOptimizer<T>"
description: "Interface for optimizing a scalar-valued function over a vector of parameters."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for optimizing a scalar-valued function over a vector of parameters.

## For Beginners

A function optimizer finds the parameter values that minimize
a given function. For example, finding the weights that minimize a loss function.
Different optimizers use different strategies: gradient descent follows the slope
downhill, L-BFGS approximates the curvature for faster convergence, etc.

## How It Works

This interface abstracts the optimization strategy so algorithms can swap
optimizers without changing their structure. For example, NOTEARS can use L-BFGS
(default) or gradient descent for experimentation.

## Methods

| Method | Summary |
|:-----|:--------|
| `Minimize(Vector<>,Func<Vector<>,ValueTuple<,Vector<>>>,Int32,)` | Minimizes a function starting from the given initial parameters. |

