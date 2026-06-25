---
title: "NeuralGARCH<T>"
description: "Neural GARCH model for forecasting asset volatility."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Volatility`

Neural GARCH model for forecasting asset volatility.

## For Beginners

GARCH is a popular statistical model for volatility.
This neural version learns the same idea from data instead of fixed formulas.
It looks at recent returns and predicts how bouncy prices will be next.

## How It Works

NeuralGARCH extends the classic GARCH idea with a neural network, allowing
non-linear relationships between recent returns and future volatility.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralGARCH(NeuralNetworkArchitecture<>,NeuralGARCHOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a NeuralGARCH model in native mode for training. |
| `NeuralGARCH(NeuralNetworkArchitecture<>,String,NeuralGARCHOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a NeuralGARCH model using a pretrained ONNX model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateRealizedVolatility(Tensor<>)` | Calculates realized volatility from high-frequency returns. |
| `ComputeCorrelationMatrix(Tensor<>)` | Computes the correlation matrix. |
| `ComputeCovarianceMatrix(Tensor<>)` | Computes the covariance matrix. |
| `CreateNewInstance` | Creates a new instance for cloning. |
| `EnsureHorizonShape(Tensor<>,Int32)` | Ensures the forecast tensor has the requested horizon. |
| `EstimateCurrentVolatility(Tensor<>)` | Estimates the current volatility from recent returns. |
| `ForecastNative(Tensor<>,Double[])` | Forecasts volatility using native layers. |
| `ForecastVolatility(Tensor<>,Int32)` | Forecasts future volatility. |
| `GetFinancialMetrics` | Gets overall financial metrics for the model. |
| `GetModelMetadata` | Returns metadata describing the model. |
| `GetOptions` |  |
| `GetReturnAt(Tensor<>,Int32,Int32,Int32)` | Reads a return value at the given sample and asset index. |
| `GetReturnMatrixShape(Tensor<>,Int32,Int32)` | Gets the sample and asset dimensions for return tensors. |
| `GetVolatilityMetrics` | Gets volatility-specific metrics. |
| `InitializeLayers` | Initializes the NeuralGARCH layers. |
| `PredictCore(Tensor<>)` | Runs a forward pass through the network. |
| `TrainCore(Tensor<>,Tensor<>,Tensor<>)` | Core training logic for NeuralGARCH. |
| `UpdateParameters(Vector<>)` | Updates model parameters from a flat vector. |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers for the NeuralGARCH model. |
| `ValidateInputShape(Tensor<>)` | Validates input shape for the volatility model. |

