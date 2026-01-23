# AiDotNet API Reference

Welcome to the AiDotNet API Reference documentation. This section provides complete API documentation for all public classes, interfaces, methods, and properties.

## Core Namespaces

| Namespace | Description |
|:----------|:------------|
| [AiDotNet](AiDotNet.yml) | Core builder and result types (AiModelBuilder, AiModelResult) |
| [AiDotNet.Configuration](AiDotNet.Configuration.yml) | Configuration options and settings |
| [AiDotNet.NeuralNetworks](AiDotNet.NeuralNetworks.yml) | 100+ neural network architectures |
| [AiDotNet.Classification](AiDotNet.Classification.yml) | 28+ classification algorithms |
| [AiDotNet.Regression](AiDotNet.Regression.yml) | 41+ regression algorithms |
| [AiDotNet.Clustering](AiDotNet.Clustering.yml) | 20+ clustering algorithms |
| [AiDotNet.ComputerVision](AiDotNet.ComputerVision.yml) | 50+ vision models (YOLO, DETR, SAM) |
| [AiDotNet.Audio](AiDotNet.Audio.yml) | 90+ audio models (Whisper, TTS) |
| [AiDotNet.ReinforcementLearning](AiDotNet.ReinforcementLearning.yml) | 80+ RL agents (DQN, PPO, SAC) |
| [AiDotNet.Diffusion](AiDotNet.Diffusion.yml) | 20+ diffusion models |
| [AiDotNet.LoRA](AiDotNet.LoRA.yml) | 37+ LoRA adapters (QLoRA, DoRA) |
| [AiDotNet.RAG](AiDotNet.RAG.yml) | 50+ RAG components |
| [AiDotNet.DistributedTraining](AiDotNet.DistributedTraining.yml) | DDP, FSDP, ZeRO strategies |
| [AiDotNet.AutoML](AiDotNet.AutoML.yml) | Automatic model selection |
| [AiDotNet.Serving](AiDotNet.Serving.yml) | Production model serving |
| [AiDotNet.Tensors](AiDotNet.Tensors.yml) | Tensor operations and linear algebra |
| [AiDotNet.Tokenization](AiDotNet.Tokenization.yml) | Text tokenization (BPE, WordPiece) |

## Quick Start

```csharp
using AiDotNet;

// Build and train a model using the facade pattern
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new NeuralNetwork<double>(inputSize: 10, hiddenSize: 64, outputSize: 2))
    .ConfigureOptimizer(new AdamOptimizer<double>())
    .ConfigurePreprocessing()
    .BuildAsync(features, labels);

// Make predictions using the result directly
var prediction = result.Predict(newSample);
```

## Entry Points

### AiModelBuilder

The `AiModelBuilder<T, TInput, TOutput>` class is your primary entry point for building and training models. It uses a fluent builder pattern for configuration.

Key methods:
- `ConfigureModel()` - Set the model architecture
- `ConfigureOptimizer()` - Set the training optimizer
- `ConfigurePreprocessing()` - Configure data preprocessing
- `ConfigureAutoML()` - Enable automatic model selection
- `ConfigureHuggingFace()` - Load HuggingFace models
- `ConfigureDistributedTraining()` - Enable multi-GPU training
- `BuildAsync()` - Build and train the model

### AiModelResult

The `AiModelResult<T, TInput, TOutput>` class wraps your trained model and provides inference capabilities.

Key properties and methods:
- `Predict()` - Make predictions on new data
- `Model` - Access the underlying trained model
- `Metrics` - Training and validation metrics
- `Save()` / `Load()` - Model persistence

## See Also

- [Getting Started Guide](../docs/getting-started/index.md)
- [Tutorials](../docs/tutorials/index.md)
- [Samples Repository](https://github.com/ooples/AiDotNet/tree/master/samples)
