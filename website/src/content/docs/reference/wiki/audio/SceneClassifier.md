---
title: "SceneClassifier<T>"
description: "Acoustic scene classification model for identifying recording environments."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Classification`

Acoustic scene classification model for identifying recording environments.

## For Beginners

Scene classification answers "Where was this recorded?":

- Indoor: office, home, shopping mall, restaurant, library
- Outdoor: street, park, beach, forest, construction site
- Transportation: bus, train, metro, airport, car

Usage with ONNX model:

Usage for training:

## How It Works

Classifies audio recordings by their acoustic environment or scene context.
Based on DCASE (Detection and Classification of Acoustic Scenes and Events) challenge.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SceneClassifier(NeuralNetworkArchitecture<>,SceneClassifierOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a SceneClassifier for native training mode. |
| `SceneClassifier(SceneClassifierOptions)` | Creates a SceneClassifier with default options for basic classification. |
| `SceneClassifier(String,SceneClassifierOptions)` | Creates a SceneClassifier for ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MinimumDurationSeconds` | Gets the minimum audio duration required for reliable classification. |
| `Scenes` | Gets the scenes (alias for SupportedScenes for legacy API compatibility). |
| `SupportedScenes` | Gets the list of scenes this model can classify. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Classify(Tensor<>)` | Classifies the acoustic scene of an audio recording. |
| `ClassifyAsync(Tensor<>,CancellationToken)` | Classifies the acoustic scene asynchronously. |
| `ClassifyCategory(Tensor<>)` | Classifies audio and returns category with confidence (legacy API compatibility). |
| `CreateAsync(SceneClassifierOptions,IProgress<Double>,CancellationToken)` | Creates a SceneClassifier asynchronously with model download. |
| `CreateNewInstance` | Creates a new instance of this network type. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data. |
| `Dispose(Boolean)` | Disposes of managed resources. |
| `ExtractAcousticFeatures(Tensor<>)` | Extracts acoustic features used for scene classification. |
| `GetModelMetadata` | Gets model metadata for serialization. |
| `GetOptions` |  |
| `GetSceneProbabilities(Tensor<>)` | Gets scene probabilities for all supported scenes. |
| `GetTopScenes(Tensor<>,Int32)` | Gets top-K scene predictions. |
| `InitializeLayers` | Initializes the neural network layers. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output into final predictions. |
| `PredictCore(Tensor<>)` | Predicts scene probabilities from audio features. |
| `PreprocessAudio(Tensor<>)` | Preprocesses raw audio into model input format. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data. |
| `TrackSceneChanges(Tensor<>,Double)` | Tracks scene changes over time in longer audio. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on labeled audio samples. |
| `UpdateParameters(Vector<>)` | Updates parameters from a flattened parameter vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `SceneCategories` | Scene category mapping. |
| `StandardScenes` | Standard acoustic scene labels (DCASE-style). |

