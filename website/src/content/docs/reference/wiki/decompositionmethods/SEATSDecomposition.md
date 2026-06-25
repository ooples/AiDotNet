---
title: "SEATSDecomposition<T>"
description: "Implements the SEATS (Seasonal Extraction in ARIMA Time Series) decomposition method for time series data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.TimeSeriesDecomposition`

Implements the SEATS (Seasonal Extraction in ARIMA Time Series) decomposition method for time series data.

## For Beginners

SEATS is an advanced method for breaking down time series data into different components:
trend (long-term direction), seasonal patterns (regular fluctuations), and irregular components (random noise).
It uses statistical models to identify these patterns in your data.

## How It Works

This implementation supports multiple algorithm variants: Standard, Canonical, and Burman.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SEATSDecomposition(Vector<>,SARIMAOptions<>,SEATSAlgorithmType,Int32)` | Initializes a new instance of the `SEATSDecomposition` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyFilter(Vector<>,Vector<>)` | Applies a filter to a signal using convolution. |
| `CalculateSeasonalSpectralDensity(,Vector<>,Vector<>,Int32)` | Calculates the spectral density at a given frequency for seasonal components. |
| `CalculateSpectralDensitiesAtSeasonalFrequencies(Vector<>,Vector<>,Int32)` | Calculates the spectral densities at seasonal frequencies for the given seasonal AR and MA parameters. |
| `CalculateSpectralDensity(,Vector<>,Vector<>)` | Calculates the spectral density at a given frequency for the provided AR and MA parameters. |
| `CalculateSpectralDensityAtZero(Vector<>,Vector<>)` | Calculates the spectral density at frequency zero for the given AR and MA parameters. |
| `Decompose` | Performs the time series decomposition based on the selected algorithm. |
| `DecomposeBurman` | Performs Burman's variant of SEATS decomposition. |
| `DecomposeCanonical` | Performs canonical SEATS decomposition. |
| `DecomposeStandard` | Performs standard SEATS decomposition. |
| `DesignBurmanSeasonalFilter(SARIMAModel<>)` | Designs a Burman seasonal filter based on the provided SARIMA model. |
| `DesignBurmanTrendFilter(SARIMAModel<>)` | Designs a Burman trend filter based on the provided SARIMA model. |
| `DesignCanonicalSeasonalFilter(SARIMAModel<>)` | Designs a canonical seasonal filter based on the provided SARIMA model. |
| `DesignCanonicalTrendFilter(SARIMAModel<>)` | Designs a canonical trend filter based on the provided SARIMA model. |
| `ExtractBurmanIrregularComponent(SARIMAModel<>,Vector<>,Vector<>)` | Extracts the irregular component as the residual after removing trend and seasonal components. |
| `ExtractBurmanSeasonalComponent(SARIMAModel<>)` | Extracts the seasonal component using Burman's method. |
| `ExtractBurmanTrendComponent(SARIMAModel<>)` | Extracts the trend component using Burman's method. |
| `ExtractCanonicalIrregularComponent(SARIMAModel<>,Vector<>,Vector<>)` | Extracts the irregular component as the residual after removing trend and seasonal components. |
| `ExtractCanonicalSeasonalComponent(SARIMAModel<>)` | Extracts the seasonal component using canonical decomposition. |
| `ExtractCanonicalTrendComponent(SARIMAModel<>)` | Extracts the trend component using canonical decomposition. |
| `ExtractIrregularComponent(Vector<>,Vector<>)` | Extracts the irregular component from a time series by removing trend and seasonal components. |
| `ExtractSeasonalComponent(SARIMAModel<>)` | Extracts the seasonal component from a time series using the SARIMA model. |
| `ExtractTrendComponent(SARIMAModel<>)` | Extracts the trend component from a time series using the SARIMA model. |
| `NumericIntegration(Func<,>,,,Int32)` | Performs numerical integration of a function over a specified interval. |

