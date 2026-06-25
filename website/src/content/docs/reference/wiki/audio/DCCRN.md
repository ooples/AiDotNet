---
title: "DCCRN<T>"
description: "DCCRN - Deep Complex Convolution Recurrent Network for speech enhancement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Enhancement`

DCCRN - Deep Complex Convolution Recurrent Network for speech enhancement.

## For Beginners

DCCRN is a neural network designed specifically for
cleaning up noisy audio. Unlike simpler methods that only work with the
"loudness" of frequencies, DCCRN also considers the "timing" (phase),
which results in more natural-sounding enhanced audio.

Think of it like this: regular enhancement is like adjusting volume
of different frequencies, while DCCRN can also adjust the timing of
sound waves to better reconstruct the original clean speech.

Usage:

## How It Works

DCCRN operates directly on complex-valued spectrograms, preserving phase information
for high-quality speech enhancement. Key features:

- Complex-valued convolutions for better spectral modeling
- LSTM layers for temporal dependencies
- Skip connections for gradient flow
- Mask-based enhancement for clean speech estimation

Reference: "DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware
Speech Enhancement" by Hu et al., Interspeech 2020

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DCCRN(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Boolean,Int32,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DCCRNOptions)` | Creates a DCCRN model for native training and inference. |
| `DCCRN(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,OnnxModelOptions,DCCRNOptions)` | Creates a DCCRN model for ONNX inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EnhancementStrength` |  |
| `LatencySamples` |  |
| `NumChannels` |  |
| `NumStages` | Gets the number of encoder/decoder stages. |
| `SupportsTraining` | Gets whether this network supports training. |
| `UseComplexMask` | Gets whether complex mask is used. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyComplexMask(Tensor<>,Tensor<>)` | Applies complex mask to STFT. |
| `ComputeComplexSTFT(Tensor<>)` | Computes complex STFT. |
| `ComputeInverseSTFT(Tensor<>)` | Computes inverse STFT. |
| `ConcatenateChannels(Tensor<>,Tensor<>)` | Concatenates tensors along channel dimension with dimension validation. |
| `CreateHannWindow(Int32)` | Creates a Hann window. |
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
| `ResetState` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_baseChannels` | Base number of channels. |
| `_decoder` | Decoder layers (complex transposed convolutions). |
| `_encoder` | Encoder layers (complex convolutions). |
| `_encoderOutputDim` | Encoder output dimension (channels * freqBins) for LSTM projection. |
| `_encoderOutputs` | Cached encoder outputs for skip connections. |
| `_fftSize` | FFT size for STFT. |
| `_hopSize` | Hop size for STFT. |
| `_kernelSize` | Kernel size for convolutions. |
| `_lossFunction` | Loss function for training. |
| `_lstmHiddenDim` | LSTM hidden dimension. |
| `_lstmLayers` | LSTM layers. |
| `_lstmProjection` | Projection layer to map LSTM output back to encoder spatial dimensions. |
| `_maskLayer` | Mask estimation layer. |
| `_numLstmLayers` | Number of LSTM layers. |
| `_numStages` | Number of encoder/decoder stages. |
| `_optimizer` | Optimizer for training. |
| `_skipLayers` | Skip connection layers. |
| `_stride` | Stride for convolutions. |
| `_useComplexMask` | Whether to use complex mask estimation. |

