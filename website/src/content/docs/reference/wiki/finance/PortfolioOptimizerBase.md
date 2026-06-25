---
title: "PortfolioOptimizerBase<T>"
description: "Base class for portfolio optimization models."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Finance.Base`

Base class for portfolio optimization models.

## For Beginners

This is the foundation for models that manage investment portfolios.
It provides the basic tools needed to decide how much money to put into different assets
(stocks, bonds, etc.) to achieve the best balance of risk and return.

## How It Works

This abstract base class implements the `IPortfolioOptimizer` interface
and provides common functionality for neural portfolio optimization. It handles
asset tracking, metric calculation, and integration with the financial model hierarchy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PortfolioOptimizerBase(NeuralNetworkArchitecture<>,Int32,Int32,ILossFunction<>)` | Initializes a new instance of the PortfolioOptimizerBase class for training. |
| `PortfolioOptimizerBase(NeuralNetworkArchitecture<>,String,Int32,Int32)` | Initializes a new instance of the PortfolioOptimizerBase class from a pretrained ONNX model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AnalyticOptimizer` | Closed-form analytic mean-variance optimizer available as a training-free baseline / warm start alongside the learned weights. |
| `NumAssets` | Gets the number of assets in the portfolio universe. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateExpectedReturn(Tensor<>,Tensor<>)` | Calculates the expected return of the portfolio. |
| `CalculateSharpeRatio(Tensor<>,Tensor<>,Tensor<>,)` | Calculates the Sharpe Ratio of the portfolio. |
| `CalculateVolatility(Tensor<>,Tensor<>)` | Calculates the volatility (standard deviation) of the portfolio. |
| `ComputeRiskContribution(Tensor<>,Tensor<>)` | Computes the risk contribution of each asset to the total portfolio risk. |
| `DeserializeModelSpecificData(BinaryReader)` | Deserializes portfolio-specific model data. |
| `ForecastNative(Tensor<>,Double[])` | Generates a forecast using the native model. |
| `GetFinancialMetrics` | Gets overall financial metrics for the model. |
| `GetPortfolioMetrics` | Gets metrics for portfolio optimizer evaluation. |
| `OptimizePortfolio(Tensor<>)` | Optimizes the portfolio to determine the best asset allocation. |
| `OptimizeWeights(Tensor<>,Tensor<>)` | Calculates optimal weights given expected returns and covariance (Traditional optimization). |
| `SerializeModelSpecificData(BinaryWriter)` | Serializes portfolio-specific model data. |
| `TrainCore(Tensor<>,Tensor<>,Tensor<>)` | Core training logic for the portfolio optimizer. |
| `ValidateInputShape(Tensor<>)` | Validates the input tensor shape. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numAssets` | The number of assets in the portfolio universe. |

