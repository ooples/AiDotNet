---
title: "AudioVisualCorrespondenceNetwork<T>"
description: "Audio-visual correspondence learning network for cross-modal understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Audio-visual correspondence learning network for cross-modal understanding.

## For Beginners

This model learns to connect what it "hears" with what it "sees."

For example, if you show it a video of someone playing guitar, it learns that the
guitar sound corresponds to the guitar in the image. This enables:

- Sound source localization: "Where in the image is the sound coming from?"
- Audio-visual retrieval: "Find images that match this sound"
- Scene understanding: "What objects are making sounds in this scene?"

The network processes audio and video through separate encoders, then learns to
align them in a shared embedding space using contrastive learning.

## How It Works

This network learns correspondences between audio and visual modalities,
enabling sound source localization, audio-visual retrieval, and scene understanding.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioVisualCorrespondenceNetwork` | Initializes a new instance with default architecture settings. |
| `AudioVisualCorrespondenceNetwork(NeuralNetworkArchitecture<>,Int32,Int32,Double,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,AudioVisualCorrespondenceOptions)` | Creates a new audio-visual correspondence network. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AudioSampleRate` |  |
| `EmbeddingDimension` |  |
| `ParameterCount` |  |
| `VideoFrameRate` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CheckSynchronization(Tensor<>,IEnumerable<Tensor<>>)` |  |
| `ClassifyScene(Tensor<>,IEnumerable<Tensor<>>,IEnumerable<String>)` |  |
| `ComputeCorrespondence(Tensor<>,IEnumerable<Tensor<>>)` |  |
| `CreateNewInstance` |  |
| `DeepCopy` |  |
| `DescribeExpectedAudio(IEnumerable<Tensor<>>)` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `GetAudioEmbedding(Tensor<>,Int32)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `GetVisualEmbedding(IEnumerable<Tensor<>>)` |  |
| `InitializeLayers` |  |
| `LearnCorrespondence(IEnumerable<Tensor<>>,IEnumerable<IEnumerable<Tensor<>>>,Int32)` |  |
| `LocalizeSoundSource(Tensor<>,IEnumerable<Tensor<>>)` |  |
| `PredictCore(Tensor<>)` |  |
| `RetrieveAudioFromVisuals(IEnumerable<Tensor<>>,IEnumerable<Vector<>>,Int32)` |  |
| `RetrieveVisualsFromAudio(Tensor<>,IEnumerable<Vector<>>,Int32)` |  |
| `SeparateAudioByVisual(Tensor<>,Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SetParameters(Vector<>)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

