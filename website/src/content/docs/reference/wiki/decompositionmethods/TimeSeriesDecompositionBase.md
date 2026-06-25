---
title: "TimeSeriesDecompositionBase<T>"
description: "Base class for time series decomposition algorithms that break down time series data into component parts."
section: "API Reference"
---

`Base Classes` · `AiDotNet.DecompositionMethods.TimeSeriesDecomposition`

Base class for time series decomposition algorithms that break down time series data into component parts.

## For Beginners

Time series decomposition is like taking apart a complex toy to see its individual pieces.
It breaks down a sequence of data points (like daily sales, monthly temperatures, etc.) into simpler
components that are easier to understand. Common components include:

- Trend: The long-term direction (going up, down, or staying flat)
- Seasonal: Regular patterns that repeat (like higher sales during holidays)
- Residual: The leftover "noise" after removing trend and seasonal patterns

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeSeriesDecompositionBase(Vector<>)` | Initializes a new instance of the time series decomposition class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Components` | Dictionary storing the different components extracted from the time series. |
| `Engine` | Gets the global execution engine for vector operations. |
| `TimeSeries` | The original time series data to be decomposed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddComponent(DecompositionComponentType,Object)` | Adds a component to the decomposition results. |
| `CalculateResidual(Vector<>,Vector<>)` | Calculates the residual component by subtracting trend and seasonal components from the original time series. |
| `Decompose` | Performs the decomposition of the time series into its components. |
| `GetComponent(DecompositionComponentType)` | Retrieves a specific component from the decomposition results. |
| `GetComponentAsMatrix(DecompositionComponentType)` | Retrieves a specific component as a matrix. |
| `GetComponentAsVector(DecompositionComponentType)` | Retrieves a specific component as a vector. |
| `GetComponents` | Returns all components extracted from the time series. |
| `HasComponent(DecompositionComponentType)` | Checks if a specific component exists in the decomposition results. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides mathematical operations for the numeric type T. |

