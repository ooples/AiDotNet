---
title: "NeuralNoiseReducer<T>"
description: "Neural network-based noise reducer for high-quality audio enhancement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Enhancement`

Neural network-based noise reducer for high-quality audio enhancement.

## For Beginners

This is like a "magic eraser" for audio noise!

How it works:

1. Converts audio to a spectrogram (picture of sound)
2. Neural network learns to identify and remove noise patterns
3. Converts cleaned spectrogram back to audio

Key features:

- Works on any type of noise (AC hum, fan noise, traffic, etc.)
- Preserves speech/music quality while removing noise
- Can be trained on your specific noise conditions
- Supports real-time streaming for live applications

Use cases:

- Podcast/video production (remove background noise)
- Voice calls (improve speech clarity)
- Music restoration (remove hiss/crackle from old recordings)
- Hearing aids (enhance speech in noisy environments)

Two modes of operation:

1. ONNX Mode: Load a pre-trained model for instant use
2. Native Mode: Train your own model on custom data

## How It Works

This model uses an encoder-bottleneck-decoder architecture inspired by U-Net
to learn the mapping from noisy audio to clean audio. It operates in the
time-frequency domain using STFT for analysis and synthesis.
Note: The current implementation uses a simplified dense decoder; a full U-Net
with transposed convolutions and skip connections is planned for future versions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralNoiseReducer` | Creates a NeuralNoiseReducer with default configuration for native training mode. |
| `NeuralNoiseReducer(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,ILossFunction<>,NeuralNoiseReducerOptions)` | Creates a NeuralNoiseReducer in native training mode. |
| `NeuralNoiseReducer(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32,Double,NeuralNoiseReducerOptions)` | Creates a NeuralNoiseReducer in ONNX inference mode using a pre-trained model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EnhancementStrength` |  |
| `LatencySamples` |  |
| `NumChannels` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Interfaces#IAudioEnhancer{T}#ResetState` | Resets streaming state (explicit interface implementation to avoid conflict with base). |
| `ComputeISTFT([],[])` | Computes Inverse Short-Time Fourier Transform using FftSharp library. |
| `ComputeSTFT([])` | Computes Short-Time Fourier Transform using FftSharp library (O(N log N) algorithm). |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Enhance(Tensor<>)` |  |
| `EnhanceSpectrum([])` | Applies neural network enhancement to spectrum. |
| `EnhanceWithReference(Tensor<>,Tensor<>)` |  |
| `EstimateNoiseProfile(Tensor<>)` |  |
| `EstimateNoiseSpectrum([])` | Estimates noise spectrum from noise-only audio. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `IsMonoVectorLike(Tensor<>)` | Returns `true` when `tensor` is a mono vector — shape [N], [1, N], or [N, 1] — i.e. |
| `NLMSEchoCancel(Double[],Double[],Int32,Double,Double)` | Normalised LMS (NLMS) acoustic echo canceller. |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessAudio(Tensor<>)` |  |
| `ProcessChunk(Tensor<>)` |  |
| `ProcessOverlapAdd([])` | Processes audio using overlap-add STFT method. |
| `ProcessStreamingChunk([])` | Processes audio in streaming mode. |
| `ResetEnhancerState` | Resets the enhancer streaming state. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_baseFilters` | Base number of filters (doubled at each stage) (non-readonly for deserialization support). |
| `_bottleneckDim` | Hidden dimension in bottleneck (non-readonly for deserialization support). |
| `_bottleneckLayer` | Bottleneck layer. |
| `_bufferPosition` | Current position in input buffer. |
| `_decoderLayers` | Decoder layers (upsampling path). |
| `_encoderLayers` | Encoder layers (downsampling path). |
| `_fftSize` | FFT size for STFT analysis (non-readonly for deserialization support). |
| `_hopSize` | Hop size between STFT frames (non-readonly for deserialization support). |
| `_inputBuffer` | Input buffer for streaming mode. |
| `_lossFunction` | Loss function for training. |
| `_modelPath` | Path to ONNX model (ONNX mode only). |
| `_noiseProfile` | Noise profile estimate (optional). |
| `_numStages` | Number of encoder/decoder stages (non-readonly for deserialization support). |
| `_outputBuffer` | Output buffer for overlap-add. |
| `_outputLayer` | Output projection layer. |
| `_useNativeMode` | Indicates whether to use native training mode. |
| `_window` | Window function for STFT. |

