---
title: "RiskModelBase<T>"
description: "Base class for risk management models, providing common infrastructure for risk assessment."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Finance.Base`

Base class for risk management models, providing common infrastructure for risk assessment.

## For Beginners

This is the "parent" for all risk models. It handles the boring stuff
(like saving settings to a file or checking input shapes) so that specific models (like NeuralVaR)
can focus on the actual math of predicting risk.

## How It Works

This abstract base class implements the `IRiskModel` interface and provides the
foundation for neural network-based risk models. It handles common tasks like model configuration,
serialization, and basic metric tracking.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RiskModelBase(NeuralNetworkArchitecture<>,Int32,Double,Int32,ILossFunction<>)` | Initializes a new instance of the RiskModelBase class for training. |
| `RiskModelBase(NeuralNetworkArchitecture<>,String,Int32,Double,Int32)` | Initializes a new instance of the RiskModelBase class from a pretrained ONNX model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConfidenceLevel` | Gets the confidence level for risk calculations. |
| `RiskRatioCalculator` | The risk-adjusted-performance calculator used by `Int32)`. |
| `TimeHorizon` | Gets the time horizon for risk calculations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustForRisk(Tensor<>,)` | Adjusts a proposed action to satisfy risk constraints. |
| `CalculateCVaR(Tensor<>,Tensor<>)` | Calculates Conditional Value at Risk (CVaR). |
| `CalculateRisk(Tensor<>)` | Calculates the primary risk metric for the given input. |
| `CalculateVaR(Tensor<>,Tensor<>)` | Calculates Value at Risk (VaR). |
| `ComputeRiskRatios(IReadOnlyList<>,Double,Int32)` | Scores a realized periodic return series with the configured `RiskRatioCalculator`, returning the annualized Sharpe, Sortino, and Calmar ratios. |
| `DecomposeRisk(Tensor<>,Tensor<>)` | Decomposes total risk into contributions from individual assets. |
| `DeserializeModelSpecificData(BinaryReader)` | Deserializes risk-specific model data. |
| `EstimateExceedanceProbability(Tensor<>,Tensor<>,)` | Estimates the probability of losses exceeding a threshold. |
| `ForecastNative(Tensor<>,Double[])` | Generates a forecast using the native model. |
| `GetFinancialMetrics` | Gets overall financial metrics for the model. |
| `GetRiskMetrics` | Gets metrics for risk model evaluation. |
| `SerializeModelSpecificData(BinaryWriter)` | Serializes risk-specific model data. |
| `StressTest(Tensor<>,Tensor<>)` | Performs stress testing under specific scenarios. |
| `ValidateInputShape(Tensor<>)` | Validates the input tensor shape. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_confidenceLevel` | The confidence level used for risk calculations (e.g., 0.95 or 0.99). |
| `_timeHorizon` | The time horizon for risk calculations in days. |

