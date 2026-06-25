---
title: "MusicSourceSeparator<T>"
description: "Music source separation model for separating audio into stems (vocals, drums, bass, other)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.SourceSeparation`

Music source separation model for separating audio into stems (vocals, drums, bass, other).

## For Beginners

Source separation is like unmixing a smoothie:

- Input: Mixed audio with multiple instruments and vocals
- Output: Separate tracks for vocals, drums, bass, and other instruments
- Uses neural networks to predict which parts of the spectrum belong to each source

Usage with ONNX model:

Usage for training:

## How It Works

This implements a U-Net based source separation approach similar to Spleeter/Demucs.
The model separates mixed audio into individual instrument stems using spectral masking.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MusicSourceSeparator(NeuralNetworkArchitecture<>,SourceSeparationOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a MusicSourceSeparator for native training mode. |
| `MusicSourceSeparator(SourceSeparationOptions)` | Creates a MusicSourceSeparator for CPU-based spectral processing. |
| `MusicSourceSeparator(String,SourceSeparationOptions)` | Creates a MusicSourceSeparator for ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumStems` | Gets the number of stems/sources this model produces. |
| `SupportedSources` | Gets the sources this model can separate. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateAsync(SourceSeparationOptions,IProgress<Double>,CancellationToken)` | Creates a MusicSourceSeparator asynchronously, downloading models if needed. |
| `CreateCpuOnly(SourceSeparationOptions)` | Creates a MusicSourceSeparator for CPU-based spectral processing without neural network. |
| `CreateNewInstance` | Creates a new instance of this network type. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data — the inverse of `BinaryWriter)`. |
| `Dispose(Boolean)` | Disposes of managed resources. |
| `ExtractSource(Tensor<>,String)` | Extracts a specific source from the mix. |
| `GetModelMetadata` | Gets model metadata for serialization. |
| `GetOptions` |  |
| `GetSourceMask(Tensor<>,String)` | Gets the soft mask for a specific source. |
| `InitializeLayers` | Initializes the neural network layers. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output (applies sigmoid to mask values). |
| `PredictCore(Tensor<>)` | Predicts source masks from spectrogram magnitude. |
| `PreprocessAudio(Tensor<>)` | Preprocesses raw audio into spectrogram format. |
| `Remix(SourceSeparationResult<>,IReadOnlyDictionary<String,Double>)` | Remixes the separated sources with custom volumes. |
| `RemoveSource(Tensor<>,String)` | Removes a specific source from the mix. |
| `Separate(Tensor<>)` | Separates all sources from the audio mix. |
| `SeparateAsync(Tensor<>,CancellationToken)` | Separates all sources asynchronously. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on mixed audio and ground truth stems. |
| `UpdateParameters(Vector<>)` | Updates parameters from a flattened parameter vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `FiveStemSources` | Source names for 5-stem separation. |
| `StandardSources` | Standard source names for 4-stem separation. |
| `TwoStemSources` | Source names for 2-stem separation. |

