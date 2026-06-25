---
title: "LCMKernel<T>"
description: "Linear Coregionalization Model (LCM) kernel for multi-output Gaussian Processes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Linear Coregionalization Model (LCM) kernel for multi-output Gaussian Processes.

## For Beginners

The Linear Coregionalization Model (LCM) is a powerful method
for multi-output Gaussian Processes. It models the correlations between multiple
outputs using a sum of products of input kernels and output covariance matrices.

The model assumes each output is a linear combination of latent functions:
y_d(x) = Σᵢ aᵢ_d × uᵢ(x)

Where:

- y_d(x) is output d at input x
- uᵢ(x) are independent latent functions, each with their own kernel kᵢ(x, x')
- aᵢ_d are mixing coefficients

The resulting multi-output kernel is:
k((x, d), (x', d')) = Σᵢ Bᵢ[d, d'] × kᵢ(x, x')

Where Bᵢ = aᵢ × aᵢᵀ is the coregionalization matrix for component i.

Why use LCM?

1. **Flexible correlation structure**: Different components can capture different relationships
2. **Interpretability**: Each component models a specific pattern shared across outputs
3. **Efficiency**: Exploits Kronecker structure for faster computation

## How It Works

Applications:

- Multi-task learning (related prediction tasks)
- Sensor networks (multiple sensors measuring related quantities)
- Financial modeling (correlated asset returns)
- Geostatistics (co-kriging for multiple spatial variables)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LCMKernel(IKernelFunction<>[],Double[0:,0:][])` | Initializes a new LCM kernel with specified kernels and coregionalization matrices. |
| `LCMKernel(IKernelFunction<>[],Int32,Nullable<Int32>,Nullable<Int32>)` | Initializes a new LCM kernel with random coregionalization matrices. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumComponents` | Gets the number of components (latent functions). |
| `NumOutputs` | Gets the number of outputs. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the LCM kernel value between two output-input pairs. |
| `CalculateForOutputs(Vector<>,Vector<>,Int32,Int32)` | Calculates the kernel value for explicit output indices. |
| `CreateExtendedVector(Vector<>,Int32)` | Creates an extended vector for a specific output. |
| `ExtractInput(Vector<>)` | Extracts the input part from an extended vector. |
| `GetCoregMatrix(Int32)` | Gets a copy of the coregionalization matrix for a specific component. |
| `GetInputKernel(Int32)` | Gets the input kernel for a specific component. |
| `GetTotalCorrelation` | Gets the total correlation matrix across all components. |
| `SingleComponent(IKernelFunction<>,Int32,Double)` | Creates a simple LCM kernel with one component. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_coregMatrices` | The coregionalization matrices for each component. |
| `_inputKernels` | The input kernels for each component. |
| `_numComponents` | Number of components (latent functions). |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_numOutputs` | Number of outputs. |

