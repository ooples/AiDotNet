---
title: "ClassicalVolatilityModelBase<T>"
description: "Base for CLASSICAL (econometric) conditional-volatility models — GARCH and its variants — estimated by **maximum likelihood**, exactly as in their original papers, rather than by neural-network training."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Finance.Volatility`

Base for CLASSICAL (econometric) conditional-volatility models — GARCH and its variants — estimated by
**maximum likelihood**, exactly as in their original papers, rather than by neural-network training.
Mirrors how `RegressionBase` gives classical regressions the full model surface, but for the
volatility interface: it implements all of `IVolatilityModel` generically (forecast,
realized-vol, covariance/correlation, serialization, parameters) and delegates only the model-specific
conditional-variance recurrence + parameter set to the concrete subclass.

## How It Works

Estimation maximizes the Gaussian quasi-log-likelihood of the return series under the model's conditional
variance path, optimized with a derivative-free Nelder–Mead simplex over the (constrained, via transform)
parameters — the standard approach in econometric packages (e.g. `arch`, `rugarch`).

## Properties

| Property | Summary |
|:-----|:--------|
| `AiDotNet#Interfaces#IParameterizable{T,AiDotNet#Tensors#LinearAlgebra#Tensor{T},AiDotNet#Tensors#LinearAlgebra#Tensor{T}}#ParameterCount` |  |
| `DefaultLossFunction` |  |
| `FittedUnconditionalVariance` | Long-run (unconditional) variance implied by the fitted parameters — the forecast anchor. |
| `ModelName` | Human-readable model name (e.g. |
| `NumFeatures` |  |
| `NumOps` | Numeric operations for `T`. |
| `ParameterCount` | Number of free parameters. |
| `Parameters` | The fitted parameters in NATURAL (constrained) space; null until trained. |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `SupportsParameterInitialization` |  |
| `SupportsTraining` |  |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateRealizedVolatility(Tensor<>)` |  |
| `Clone` |  |
| `ComputeCorrelationMatrix(Tensor<>)` |  |
| `ComputeCovarianceMatrix(Tensor<>)` |  |
| `CreateInstance` | Factory for a fresh instance of the concrete model (for cloning). |
| `DeepCopy` |  |
| `Deserialize(Byte[])` |  |
| `Dispose` |  |
| `EstimateCurrentVolatility(Tensor<>)` |  |
| `FitReturns(IReadOnlyList<Double>)` | Fits the model to a return series by maximum likelihood (Nelder–Mead on the Gaussian QMLE). |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastAnnualizedVol(IReadOnlyList<Double>,Double)` | One-step-ahead ANNUALIZED volatility forecast = √(variance × periodsPerYear). |
| `ForecastNextVariance(IReadOnlyList<Double>)` | One-step-ahead conditional VARIANCE forecast given the observed return history. |
| `ForecastVolatility(Tensor<>,Int32)` |  |
| `GetActiveFeatureIndices` |  |
| `GetFeatureImportance` |  |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `GetVolatilityMetrics` |  |
| `InitialGuess(Double)` | A reasonable starting guess in natural space, given the sample variance. |
| `IsFeatureUsed(Int32)` |  |
| `LoadModel(String)` |  |
| `LoadState(Stream)` |  |
| `MeanReversionSpeed(Double[])` | Per-step persistence used for multi-step mean reversion (α+β for GARCH-type). |
| `NegLogLikelihood(Double[],Double[])` | Negative Gaussian log-likelihood of the return series under the conditional-variance path. |
| `NextVariance(Double,Double,Double[])` | The conditional-variance recurrence σ²_t given the previous variance and return. |
| `Predict(Tensor<>)` |  |
| `SanitizeParameters(Vector<>)` |  |
| `SaveModel(String)` |  |
| `SaveState(Stream)` |  |
| `Serialize` |  |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` |  |
| `SetParameters(Vector<>)` |  |
| `ToDoubles(Tensor<>)` | Flattens a univariate returns tensor ([n], [n,1], or [batch,n,1]) to a double array. |
| `ToNatural(Double[])` | Maps an UNCONSTRAINED optimizer vector to NATURAL parameters honoring the model's constraints (positivity, stationarity). |
| `ToUnconstrained(Double[])` | Inverse of `Double[])` — natural → unconstrained (for the initial simplex). |
| `Train(Tensor<>,Tensor<>)` |  |
| `UnconditionalVariance(Double[])` | Unconditional variance implied by the parameters (used to seed σ²_0 and long-run forecasts). |
| `WithParameters(Vector<>)` |  |

