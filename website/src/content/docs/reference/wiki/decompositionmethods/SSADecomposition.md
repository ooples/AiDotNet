---
title: "SSADecomposition<T>"
description: "Implements Singular Spectrum Analysis (SSA) for time series decomposition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.TimeSeriesDecomposition`

Implements Singular Spectrum Analysis (SSA) for time series decomposition.

## For Beginners

SSA is a technique that helps break down a time series (sequence of data points) into 
meaningful components like trends, seasonal patterns, and noise. Think of it like separating the 
ingredients of a mixed smoothie - you can identify the fruits, yogurt, and other components that were 
blended together.

## How It Works

SSA works by transforming your time series into a matrix, analyzing patterns using mathematical 
techniques, and then reconstructing the important components.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SSADecomposition(Vector<>,Int32,Int32,SSAAlgorithmType)` | Initializes a new instance of the SSA decomposition algorithm. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AssignComponents(Vector<>[])` | Assigns the reconstructed components to their respective types (trend, seasonal, residual). |
| `CreateTrajectoryMatrix` | Creates a trajectory matrix from the time series data. |
| `Decompose` | Performs the time series decomposition using the selected SSA algorithm. |
| `GroupComponents(Matrix<>,Vector<>,Matrix<>)` | Groups the decomposed components based on the SVD results. |
| `PerformSVD(Matrix<>)` | Performs Singular Value Decomposition on the trajectory matrix. |
| `PerformSequentialSSA` | Performs Sequential SSA using an iterative approach. |
| `PerformToeplitzSSA` | Performs Toeplitz SSA using the autocovariance matrix. |
| `ReconstructComponents(Matrix<>[])` | Reconstructs the time series components from the grouped components. |

