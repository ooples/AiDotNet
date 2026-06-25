---
title: "RealizedVolatilityTransformer<T>"
description: "Realized Volatility Transformer for attention-based volatility forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Volatility`

Realized Volatility Transformer for attention-based volatility forecasting.

## For Beginners

Transformers learn which past time points matter most.
This helps the model focus on recent shocks or patterns when predicting volatility.

## How It Works

This model applies transformer attention to recent returns to forecast volatility.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RealizedVolatilityTransformer(NeuralNetworkArchitecture<>,RealizedVolatilityTransformerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a RealizedVolatilityTransformer in native mode for training. |
| `RealizedVolatilityTransformer(NeuralNetworkArchitecture<>,String,RealizedVolatilityTransformerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a RealizedVolatilityTransformer using a pretrained ONNX model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateRealizedVolatility(Tensor<>)` | Calculates realized volatility from high-frequency returns. |
| `ComputeCorrelationMatrix(Tensor<>)` | Computes the correlation matrix. |
| `ComputeCovarianceMatrix(Tensor<>)` | Computes the covariance matrix. |
| `CreateNewInstance` | Creates a new instance for cloning. |
| `EnsureHorizonShape(Tensor<>,Int32)` | Ensures the forecast tensor has the requested horizon. |
| `EstimateCurrentVolatility(Tensor<>)` | Estimates the current volatility from recent returns. |
| `ForecastNative(Tensor<>,Double[])` | Forecasts volatility using native transformer layers. |
| `ForecastVolatility(Tensor<>,Int32)` | Forecasts future volatility. |
| `GetFinancialMetrics` | Gets overall financial metrics for the model. |
| `GetModelMetadata` | Returns metadata describing the model. |
| `GetOptions` |  |
| `GetReturnAt(Tensor<>,Int32,Int32,Int32)` | Reads a return value at the given sample and asset index. |
| `GetReturnMatrixShape(Tensor<>,Int32,Int32)` | Gets the sample and asset dimensions for return tensors. |
| `InitializeLayers` | Initializes transformer layers for volatility forecasting. |
| `PredictCore(Tensor<>)` | Runs a forward pass through the network. |
| `TrainCore(Tensor<>,Tensor<>,Tensor<>)` | Core training logic for the transformer. |
| `UpdateParameters(Vector<>)` | Updates model parameters from a flat vector. |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers for the transformer. |
| `ValidateInputShape(Tensor<>)` | Validates input shape for the transformer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_inGetVolatilityMetrics` | Gets volatility-specific metrics. |

