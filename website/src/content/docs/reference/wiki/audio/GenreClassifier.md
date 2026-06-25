---
title: "GenreClassifier<T>"
description: "Music genre classification model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Classification`

Music genre classification model.

## For Beginners

Genre classification analyzes audio characteristics:

- Tempo and rhythm patterns (fast/slow, complex/simple beats)
- Timbre and instrumentation (acoustic vs electronic sounds)
- Harmonic content (simple vs complex chords)
- Spectral features (brightness, warmth, texture)

Usage:

## How It Works

Classifies audio into music genres using spectral features and neural network models.
Supports common genres: rock, pop, hip-hop, jazz, classical, electronic, country, R&B, metal, folk.

This class supports both:

- **ONNX mode**: Load pre-trained models for fast inference
- **Native training mode**: Train from scratch using the layer architecture

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GenreClassifier(GenreClassifierOptions)` | Creates a new genre classifier with legacy options only. |
| `GenreClassifier(NeuralNetworkArchitecture<>,GenreClassifierOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a new genre classifier in native training mode. |
| `GenreClassifier(NeuralNetworkArchitecture<>,String,GenreClassifierOptions)` | Creates a new genre classifier in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Genres` | Gets the supported genre labels (legacy). |
| `IsOnnxMode` | Gets whether the model is operating in ONNX inference mode. |
| `SampleRate` | Gets the sample rate. |
| `SupportedGenres` | Gets the supported genre labels. |
| `SupportsMultiLabel` | Gets whether this model supports multi-label classification. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Interfaces#IGenreClassifier{T}#ExtractFeatures(Tensor<>)` | Extracts audio features used for classification. |
| `Classify(Tensor<>)` | Classifies the genre of an audio clip. |
| `ClassifyAsync(Tensor<>,CancellationToken)` | Classifies audio asynchronously. |
| `ClassifyBatch(IEnumerable<Tensor<>>)` | Classifies multiple audio segments in batch (legacy API). |
| `ClassifyLegacy(Tensor<>)` | Classifies the genre of an audio clip (legacy API). |
| `CreateAsync(GenreClassifierOptions,IProgress<Double>,CancellationToken)` | Creates a GenreClassifier asynchronously with model download. |
| `CreateNewInstance` | Creates a new instance of this model for cloning. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data. |
| `Dispose(Boolean)` | Disposes managed resources. |
| `GetGenreProbabilities(Tensor<>)` | Gets genre probabilities for all supported genres. |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetOptions` |  |
| `GetTopGenres(Tensor<>,Int32)` | Gets top-K genre predictions. |
| `InitializeLayers` | Initializes the neural network layers. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output into the final result format. |
| `PredictCore(Tensor<>)` | Predicts output for the given input. |
| `PreprocessAudio(Tensor<>)` | Preprocesses raw audio for model input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data. |
| `TrackGenreOverTime(Tensor<>,Double)` | Tracks genre over time within a piece. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on a single example. |
| `UpdateParameters(Vector<>)` | Updates model parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `StandardGenres` | Standard music genres. |

