---
title: "IndexKernel<T>"
description: "Index kernel for multi-task/multi-output Gaussian Processes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Index kernel for multi-task/multi-output Gaussian Processes.

## For Beginners

The IndexKernel enables GPs to model multiple related tasks or outputs
simultaneously, sharing information between them.

In multi-task learning, you have multiple related prediction tasks. For example:

- Predicting exam scores for multiple subjects (math, science, english)
- Forecasting sales for multiple products
- Estimating properties of multiple chemical compounds

The IndexKernel models the correlation between tasks using a covariance matrix:
k_task(t, t') = B[t, t']

Where B is a positive semi-definite "task covariance matrix" that captures:

- B[t, t]: The variance of task t
- B[t, t']: The covariance between tasks t and t'

When combined with an input kernel (e.g., RBF), the full multi-task kernel is:
k((x, t), (x', t')) = k_input(x, x') × k_task(t, t')
= k_input(x, x') × B[t, t']

This allows the GP to:

- Share information between similar tasks
- Leverage data from data-rich tasks to help data-poor tasks
- Learn task relationships from data

## How It Works

Usage patterns:

1. Initialize with a task covariance matrix (if known)
2. Initialize randomly and optimize the task covariance
3. Use with ProductKernel to combine with input kernel

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IndexKernel(Double[0:,0:])` | Initializes an IndexKernel with a specified task covariance matrix. |
| `IndexKernel(Int32,Nullable<Int32>,Double,Nullable<Int32>)` | Initializes an IndexKernel with specified number of tasks using random initialization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumTasks` | Gets the number of tasks. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the task kernel value between two task indices. |
| `CalculateForTasks(Int32,Int32)` | Gets the task covariance for explicit task indices. |
| `GetTaskCorrelation(Int32,Int32)` | Gets the correlation between two tasks. |
| `GetTaskCovariance` | Gets a copy of the task covariance matrix. |
| `Independent(Int32,Double)` | Creates an IndexKernel with identity task covariance (independent tasks). |
| `UniformCorrelation(Int32,Double,Double)` | Creates an IndexKernel with uniform task correlation. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_choleskyFactor` | Lower Cholesky factor of the task covariance (for parameterization). |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_numTasks` | Number of tasks. |
| `_taskCovariance` | The task covariance matrix (B). |

