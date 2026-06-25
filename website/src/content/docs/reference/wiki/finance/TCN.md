---
title: "TCN<T>"
description: "TCN (Temporal Convolutional Network) model for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Neural`

TCN (Temporal Convolutional Network) model for time series forecasting.

## For Beginners

TCN is a modern alternative to LSTM/GRU for sequence modeling.

**Key Innovation - Dilated Convolutions:**
Imagine reading a book. Regular convolution reads 3 consecutive words at a time.
Dilated convolution can skip words to cover more context:

- Dilation 1: Reads words 1, 2, 3
- Dilation 2: Reads words 1, 3, 5 (skipping every other word)
- Dilation 4: Reads words 1, 5, 9 (skipping 3 words)

By stacking layers with increasing dilation, TCN can "see" very far into the past
efficiently. With 8 layers and kernel size 3, it can consider over 1000 past time steps!

**Causal Convolutions:**
TCN only looks at past and present values, never future ones. This makes it suitable
for real-time prediction where you can't peek ahead.

**Residual Connections:**
Each block adds its input to its output: output = block(input) + input
This helps gradients flow during training and allows the network to learn
when to ignore certain blocks.

## How It Works

TCN uses dilated causal convolutions to model temporal sequences. It offers several advantages
over recurrent networks:

- Parallel computation (faster training)
- Flexible receptive field through dilation
- No vanishing gradient problem
- Better handling of long sequences

**Reference:** Bai et al., "An Empirical Evaluation of Generic Convolutional and
Recurrent Networks for Sequence Modeling", 2018. https://arxiv.org/abs/1803.01271

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TCN` | Creates a TCN in native mode for training from scratch. |
| `TCN(NeuralNetworkArchitecture<>,String,TCNOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TCN using pretrained ONNX model. |
| `TCN(NeuralNetworkArchitecture<>,TCNOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TCN model using the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
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
| `AddTensors(Tensor<>,Tensor<>)` | Element-wise addition of two tensors. |
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors into a single result. |
| `CreateNewInstance` | Creates a new instance of this model with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads TCN-specific configuration during deserialization. |
| `Dispose(Boolean)` | Releases managed resources used by this model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to specific layers from the layer collection. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through the TCN. |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward: routes through `Tensor{` directly so dropout between dilated conv blocks stays active during backprop. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetOrCreateBaseOptimizer` | Trains through an Adam (AMSGrad) optimizer at a reduced learning rate (1e-5 vs the 1e-3 framework default). |
| `InitializeLayers` | Initializes the neural network layers for TCN. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes TCN-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts the input tensor by incorporating new predictions. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet TCN architectural requirements. |
| `ValidateOptions(TCNOptions<>)` | Validates the TCN options. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_dropout` | Dropout rate for regularization. |
| `_forecastHorizon` | The forecast horizon. |
| `_inputProjection` | Input projection layer. |
| `_kernelSize` | Kernel size for convolutions. |
| `_lookbackWindow` | The lookback window size. |
| `_lossFunction` | The loss function for training. |
| `_numChannels` | Number of channels in each layer. |
| `_numFeatures` | Number of input features. |
| `_numLayers` | Number of TCN layers. |
| `_optimizer` | The optimizer for training. |
| `_outputProjection` | Output projection layer. |
| `_tcnBlocks` | TCN blocks organized by layer. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |
| `_useResidualConnections` | Whether to use residual connections. |

