---
layout: default
title: Reference
nav_order: 4
has_children: true
permalink: /reference/
---

# API Reference
{: .no_toc }

Complete reference documentation for all AiDotNet components.
{: .fs-6 .fw-300 }

---

## Core Namespaces

| Namespace | Description |
|:----------|:------------|
| `AiDotNet` | Core builder and result types |
| `AiDotNet.NeuralNetworks` | 100+ neural network architectures |
| `AiDotNet.Classification` | 28+ classification algorithms |
| `AiDotNet.Regression` | 41+ regression algorithms |
| `AiDotNet.Clustering` | 20+ clustering algorithms |
| `AiDotNet.ComputerVision` | 50+ computer vision models |
| `AiDotNet.Audio` | 90+ audio processing models |
| `AiDotNet.Video` | 34+ video processing models |
| `AiDotNet.ReinforcementLearning` | 80+ RL agents |
| `AiDotNet.MetaLearning` | 18+ meta-learning algorithms |
| `AiDotNet.SelfSupervisedLearning` | 10+ SSL methods |
| `AiDotNet.Diffusion` | 20+ diffusion models |
| `AiDotNet.TimeSeries` | 30+ time series models |
| `AiDotNet.RetrievalAugmentedGeneration` | 50+ RAG components |
| `AiDotNet.LoRA` | 37+ LoRA adapters |
| `AiDotNet.DistributedTraining` | 10+ distributed strategies |
| `AiDotNet.Optimizers` | 42+ optimization algorithms |
| `AiDotNet.LossFunctions` | 37+ loss functions |
| `AiDotNet.ActivationFunctions` | 41+ activation functions |

## Key Classes

### AiModelBuilder

The main entry point for building models:

```csharp
public class AiModelBuilder<T, TInput, TOutput>
```

[Full documentation →](./prediction-model-builder)

### AiModelResult

Contains the trained model and results:

```csharp
public class AiModelResult<T, TInput, TOutput>
```

[Full documentation →](./prediction-model-result)

## Configuration Classes

| Class | Purpose |
|:------|:--------|
| `GpuAccelerationConfig` | GPU training settings |
| `MixedPrecisionConfig` | FP16/FP32 training |
| `DistributedConfig` | Multi-GPU/node settings |
| `TrainingPipelineConfig` | Training pipeline options |
| `ReasoningConfig` | AI reasoning settings |

## Browse by Category

- [Neural Networks](./neural-networks) - All 100+ architectures
- [Classical ML](./classical-ml) - Classification, regression, clustering
- [Computer Vision](./computer-vision) - Object detection, segmentation, OCR
- [Audio](./audio) - Speech recognition, TTS, music
- [Reinforcement Learning](./reinforcement-learning) - All 80+ agents
- [Optimizers](./optimizers) - All 42+ optimizers
- [Loss Functions](./loss-functions) - All 37+ loss functions
