---
title: "DeepFilterNet<T>"
description: "DeepFilterNet - State-of-the-art deep filtering network for speech enhancement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Enhancement`

DeepFilterNet - State-of-the-art deep filtering network for speech enhancement.

## For Beginners

DeepFilterNet is like having an intelligent audio engineer
that can separate speech from background noise in real-time. It's particularly
effective because it processes audio the way humans perceive sound - focusing
more on frequencies that matter for understanding speech.

The model works by:

1. Converting audio to a time-frequency representation (spectrogram)
2. Applying learned filters to suppress noise while preserving speech
3. Reconstructing clean audio from the enhanced spectrogram

Usage:

## How It Works

DeepFilterNet is a hybrid time-frequency domain model that combines:

- ERB (Equivalent Rectangular Bandwidth) filterbank for perceptually-motivated processing
- Deep filtering in the complex STFT domain for fine-grained enhancement
- Efficient architecture with grouped convolutions for real-time processing

Reference: "DeepFilterNet: A Low Complexity Speech Enhancement Framework for
Full-Band Audio based on Deep Filtering" by Schröter et al., ICASSP 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepFilterNet` | Creates a DeepFilterNet model with default configuration for native training mode. |
| `DeepFilterNet(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DeepFilterNetOptions)` | Creates a DeepFilterNet model for native training and inference. |
| `DeepFilterNet(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,OnnxModelOptions,DeepFilterNetOptions)` | Creates a DeepFilterNet model for ONNX inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DfOrder` | Gets the deep filter order. |
| `EnhancementStrength` |  |
| `LatencySamples` |  |
| `NumChannels` |  |
| `NumErbBands` | Gets the number of ERB bands used. |
| `SupportsTraining` | Gets whether this network supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CombineOutputs(Tensor<>,Tensor<>)` | Combines deep filtering coefficients and gains. |
| `ComputeErbFeatures(Tensor<Complex<>>)` | Computes ERB (Equivalent Rectangular Bandwidth) features from complex STFT. |
| `ComputeSTFT(Tensor<>)` | Computes STFT of audio signal using ShortTimeFourierTransform. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` | Disposes resources. |
| `Enhance(Tensor<>)` |  |
| `EnhanceWithReference(Tensor<>,Tensor<>)` |  |
| `EstimateNoiseProfile(Tensor<>)` |  |
| `ForwardNative(Tensor<>)` | Forward pass through native layers. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `InitializeNativeLayers` | Initializes native layers for training mode. |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessAudio(Tensor<>)` |  |
| `ProcessChunk(Tensor<>)` |  |
| `ReconstructAudio(Tensor<>)` | Reconstructs audio from enhanced representation by applying gains and deep filtering to the cached STFT and performing inverse STFT. |
| `ResetState` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cachedComplexStft` | Cached complex STFT from preprocessing, used for audio reconstruction. |
| `_convKernelSize` | Convolution kernel size for feature extraction. |
| `_decoder` | Decoder layers for reconstruction. |
| `_dfBins` | Number of frequency bins to apply deep filtering. |
| `_dfLayers` | Deep filtering layers. |
| `_dfOrder` | Number of DeepFilter coefficients per frequency bin. |
| `_erbEncoder` | ERB encoder layers. |
| `_fftSize` | FFT size for STFT analysis. |
| `_gainLayer` | Gain estimation layer. |
| `_gruLayers` | GRU layers for temporal modeling. |
| `_hiddenDim` | Hidden dimension for the encoder/decoder. |
| `_hopSize` | Hop size for STFT. |
| `_lookahead` | Lookahead frames for causal processing. |
| `_lossFunction` | Loss function for training. |
| `_numErbBands` | Number of ERB (Equivalent Rectangular Bandwidth) bands. |
| `_numGruLayers` | Number of GRU layers in the enhancement network. |
| `_optimizer` | Optimizer for training. |

