---
title: "VisionTS<T>"
description: "VisionTS — Visual Masked Autoencoders as Zero-Shot Time Series Forecasters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

VisionTS — Visual Masked Autoencoders as Zero-Shot Time Series Forecasters.

## For Beginners

VisionTS takes a surprising approach: it converts time series
data into images and uses a vision model (originally trained on photos) to forecast future
values. The data is arranged in a 2D grid like pixels, and the vision model fills in the
missing parts, effectively predicting future values. This works because patterns in time
series grids resemble visual textures that vision models already understand.

## How It Works

VisionTS repurposes Visual Masked Autoencoders (MAE) pretrained on images for time series
forecasting. It converts time series into 2D image-like patch grids, processes them with
a pretrained ViT encoder, and reconstructs/forecasts using the decoder. This cross-modal
transfer demonstrates that vision foundation models generalize to time series.

**Reference:** "VisionTS: Visual Masked Autoencoders as Zero-Shot Time Series Forecasters",
ICML 2025.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VisionTS(NeuralNetworkArchitecture<>,String,VisionTSOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a VisionTS model using a pretrained ONNX model. |
| `VisionTS(NeuralNetworkArchitecture<>,VisionTSOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a VisionTS model in native mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
| `MaxContextLength` |  |
| `MaxPredictionHorizon` |  |
| `ModelSize` |  |
| `NumFeatures` |  |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` |  |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `CreateNewInstance` |  |
| `DenormalizeForecast(Tensor<>)` | RevIN reverse step (Kim et al. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `Forecast(Tensor<>,Double[])` |  |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

