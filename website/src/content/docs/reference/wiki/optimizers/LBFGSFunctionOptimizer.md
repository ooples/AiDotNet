---
title: "LBFGSFunctionOptimizer<T>"
description: "L-BFGS (Limited-memory BFGS) optimizer for minimizing a scalar function of a vector."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

L-BFGS (Limited-memory BFGS) optimizer for minimizing a scalar function of a vector.

## For Beginners

L-BFGS is a fast optimizer that uses information from recent
iterations to approximate the curvature of the objective function. This lets it take
much better steps than simple gradient descent, converging in far fewer iterations.

## How It Works

Implements the L-BFGS two-loop recursion (Nocedal and Wright, "Numerical Optimization",
Algorithm 7.4) with backtracking line search using the Armijo condition.

This implementation operates on generic `Vector` using
`INumericOperations` for all arithmetic, making it work with any
numeric type (float, double, decimal, etc.).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LBFGSFunctionOptimizer(Int32,Int32)` | Creates a new L-BFGS function optimizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Minimize(Vector<>,Func<Vector<>,ValueTuple<,Vector<>>>,Int32,)` |  |
| `TwoLoopRecursion(Vector<>,List<Vector<>>,List<Vector<>>,Int32)` | L-BFGS two-loop recursion to compute search direction. |

