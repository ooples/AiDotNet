---
title: "ConvTasNet<T>"
description: "Conv-TasNet: A fully-convolutional time-domain audio separation network."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Enhancement`

Conv-TasNet: A fully-convolutional time-domain audio separation network.

## For Beginners

Conv-TasNet is like having multiple microphones that each focus
on one speaker in a noisy room. Give it a recording with multiple people talking,
and it separates them into individual clean tracks!

Traditional methods convert audio to frequency domain, process it, then convert back.
Conv-TasNet works directly on the waveform, which avoids problems with phase reconstruction
and often produces cleaner results.

Common use cases:

- Separating speakers in meeting recordings
- Isolating vocals from music
- Removing background noise
- Speech enhancement for hearing aids
- Denoising phone calls

## How It Works

Conv-TasNet (Convolutional Time-domain Audio Separation Network) is a pioneering
neural network architecture that operates directly in the time domain, avoiding
the phase reconstruction problems of frequency-domain methods.

The architecture consists of three main components:

- Encoder: Converts waveform to a learned representation using 1D convolutions
- Separator: Temporal Convolutional Network (TCN) that estimates source masks
- Decoder: Reconstructs separated waveforms from masked representations

Reference: Luo, Y., & Mesgarani, N. (2019). Conv-TasNet: Surpassing Ideal Time-Frequency
Magnitude Masking for Speech Separation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConvTasNet(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,ConvTasNetOptions)` | Initializes a new instance of the `ConvTasNet` class for native training mode. |
| `ConvTasNet(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32,OnnxModelOptions,ConvTasNetOptions)` | Initializes a new instance of the `ConvTasNet` class for ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EncoderDimension` | Gets the encoder dimension (number of basis functions). |
| `EncoderKernelSize` | Gets the encoder kernel size (window length in samples). |
| `EnhancementStrength` |  |
| `LatencySamples` |  |
| `NumChannels` |  |
| `NumSources` | Gets the number of sources the network separates. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyMasks(Tensor<>,Tensor<>)` | Applies masks to encoder output to separate sources. |
| `BottleneckProject(Tensor<>)` | Projects to bottleneck dimension. |
| `ComputeGradients(Tensor<>,Tensor<>)` | Computes gradients for backpropagation. |
| `ComputeSiSnrLoss(Tensor<>,Tensor<>)` | Computes the SI-SNR loss for speech separation. |
| `CreateNewInstance` |  |
| `Decode(Tensor<>,Int32)` | Decodes masked representations back to waveform. |
| `Deserialize(Byte[])` | Deserializes the model state from a byte array. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Encode(Tensor<>)` | Encodes waveform using learned basis functions. |
| `Enhance(Tensor<>)` |  |
| `EnhanceWithReference(Tensor<>,Tensor<>)` |  |
| `EstimateMasks(Tensor<>)` | Estimates separation masks for each source. |
| `EstimateNoiseProfile(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers. |
| `LayerNorm(Tensor<>)` | Applies layer normalization. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output. |
| `PredictCore(Tensor<>)` | Predicts separated sources from input audio. |
| `PreprocessAudio(Tensor<>)` | Preprocesses raw audio waveform for model input. |
| `ProcessChunk(Tensor<>)` |  |
| `ResetState` |  |
| `RunTcn(Tensor<>)` | Runs the Temporal Convolutional Network. |
| `SeparateSources(Tensor<>)` | Separates audio into individual source signals. |
| `Serialize` | Serializes the model state to a byte array. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` | Trains the model on a batch of mixture-source pairs. |
| `UpdateParameters(Vector<>)` |  |
| `UpdateWeights(Dictionary<String,[]>)` | Updates model weights using computed gradients. |

