---
layout: landing
---

# AiDotNet

## The comprehensive .NET machine learning library

AiDotNet provides everything you need to build, train, and deploy machine learning models in .NET applications.

---

## Quick Navigation

| Section | Description |
|:--------|:------------|
| [Getting Started](docs/getting-started/index.md) | Installation and first steps |
| [Tutorials](docs/tutorials/index.md) | Step-by-step learning guides |
| [API Reference](api/index.md) | Complete API documentation |
| [Examples](docs/examples/index.md) | Code examples and samples |

---

## Features

- **Neural Networks**: Dense, CNN, RNN, LSTM, Transformer architectures
- **Classical ML**: Classification, Regression, Clustering, Dimensionality Reduction
- **Computer Vision**: Image classification, object detection, segmentation
- **NLP**: Text classification, embeddings, RAG pipelines
- **Audio**: Speech recognition (Whisper), TTS, speaker diarization
- **Time Series**: Forecasting, anomaly detection
- **GPU Acceleration**: CUDA, OpenCL, Metal support
- **Cross-Platform**: Windows, Linux, macOS

---

## Installation

```bash
dotnet add package AiDotNet
```

---

## Quick Example

```csharp
using AiDotNet;
using AiDotNet.Classification;

// Train a classifier
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new RandomForestClassifier<double>(nEstimators: 100))
    .ConfigurePreprocessing()
    .ConfigureCrossValidation(new KFoldCrossValidator<double>(k: 5))
    .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(features, labels))
    .BuildAsync();

// Make predictions
var prediction = result.Predict(newSample);
```

---

## Try It Online

Check out the [Interactive Playground](playground/) to experiment with AiDotNet directly in your browser.

---

## Links

- [GitHub Repository](https://github.com/ooples/AiDotNet)
- [NuGet Package](https://www.nuget.org/packages/AiDotNet)
- [Report Issues](https://github.com/ooples/AiDotNet/issues)
