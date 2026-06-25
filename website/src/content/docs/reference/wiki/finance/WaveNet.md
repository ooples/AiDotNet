---
title: "WaveNet<T>"
description: "WaveNet model adapted for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Neural`

WaveNet model adapted for time series forecasting.

## For Beginners

WaveNet is like a more sophisticated version of TCN:

**Gated Activation Units:**
Instead of using simple ReLU activations, WaveNet multiplies two signals:

- output = tanh(filter_output) * sigmoid(gate_output)
- The tanh extracts features, the sigmoid controls which features pass through
- This is inspired by LSTM gates and helps model complex patterns

**Two Types of Connections:**

1. **Residual Connection:** Each block adds its input to output (like TCN)
- Helps gradients flow during training
2. **Skip Connection:** Each block sends output directly to the final layers
- Allows combining features from ALL time scales
- Skip outputs are summed at the end

**Why Two Connection Types?**
The residual path maintains the signal through deep networks.
The skip path lets early layers contribute directly to the output,
ensuring fine-grained (high-frequency) and coarse (low-frequency) patterns
are both captured in the final prediction.

**Example:**
With 2 stacks of 8 layers each:

- 16 total dilated layers
- Dilations: [1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128]
- Receptive field: 2 * (2^8 - 1) + 1 = 511 time steps

## How It Works

WaveNet was originally developed by DeepMind for audio generation. It uses dilated causal
convolutions with gated activation units and dual residual/skip connections. This architecture
has proven highly effective for time series forecasting as well.

**Reference:** van den Oord et al., "WaveNet: A Generative Model for Raw Audio", 2016.
https://arxiv.org/abs/1609.03499

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WaveNet(NeuralNetworkArchitecture<>,String,WaveNetOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a WaveNet using pretrained ONNX model. |
| `WaveNet(NeuralNetworkArchitecture<>,WaveNetOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a WaveNet in native mode for training from scratch. |

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
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads WaveNet-specific configuration during deserialization. |
| `Dispose(Boolean)` | Releases managed resources used by this model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to specific layers from the layer collection. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through WaveNet. |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward: routes through `Tensor{` directly so WaveNet's dropout between dilated causal-conv blocks stays active during backprop. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for WaveNet. |
| `MultiplyTensors(Tensor<>,Tensor<>)` | Element-wise multiplication of two tensors (for gating). |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes WaveNet-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts the input tensor by incorporating new predictions. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet WaveNet architectural requirements. |
| `ValidateOptions(WaveNetOptions<>)` | Validates the WaveNet options. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_inputProjection` | Input projection layer. |
| `_output1` | First output layer after skip aggregation. |
| `_output2` | Final output projection layer. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |
| `_waveNetBlocks` | WaveNet blocks organized by layer. |

