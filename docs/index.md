---
layout: default
title: Home
nav_order: 1
description: "AiDotNet - The most comprehensive AI/ML framework for .NET"
permalink: /
---

# AiDotNet Documentation
{: .fs-9 }

The most comprehensive AI/ML framework for .NET with 4,300+ implementations across 60+ feature categories.
{: .fs-6 .fw-300 }

[Get Started](/getting-started/){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/ooples/AiDotNet){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Feature Highlights

| Category | Count | Key Features |
|:---------|:------|:-------------|
| Neural Networks | 100+ | CNN, RNN, Transformer, GAN, VAE, GNN |
| Classical ML | 106+ | Classification, Regression, Clustering |
| Computer Vision | 50+ | YOLO v8-11, DETR, Mask R-CNN, OCR |
| Audio Processing | 90+ | Whisper, TTS, Music Generation |
| Reinforcement Learning | 80+ | DQN, PPO, SAC, Multi-Agent |
| RAG & Embeddings | 50+ | Vector stores, Retrievers, Rerankers |
| LoRA Fine-tuning | 37+ | QLoRA, DoRA, AdaLoRA |
| Distributed Training | 10+ | DDP, FSDP, ZeRO |

## Quick Start

### Installation

```bash
dotnet add package AiDotNet
```

### Hello World

```csharp
using AiDotNet;

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new NeuralNetwork<double>(inputSize: 4, hiddenSize: 16, outputSize: 3))
    .ConfigureOptimizer(new AdamOptimizer<double>())
    .ConfigurePreprocessing()
    .BuildAsync(features, labels);

// Use result.Predict() directly - this is the facade pattern
var prediction = result.Predict(newSample);
```

## What do you want to build?

<div class="grid-container">

| Task | Documentation |
|:-----|:--------------|
| **Classify data** | [Classification Tutorial](/tutorials/classification/) |
| **Detect objects** | [Computer Vision Tutorial](/tutorials/computer-vision/) |
| **Process text & RAG** | [NLP & RAG Tutorial](/tutorials/nlp/) |
| **Fine-tune LLMs** | [LLM Fine-tuning Tutorial](/tutorials/llm-fine-tuning/) |
| **Train RL agents** | [Reinforcement Learning Tutorial](/tutorials/reinforcement-learning/) |
| **Scale training** | [Distributed Training Tutorial](/tutorials/distributed-training/) |
| **Deploy models** | [Deployment Tutorial](/tutorials/deployment/) |

</div>

## Why AiDotNet?

AiDotNet is the most feature-complete AI/ML framework for .NET, designed to match and exceed the capabilities of Python frameworks while providing native .NET performance and developer experience.

### Comprehensive Framework Comparison

| Feature | AiDotNet | TorchSharp | TensorFlow.NET | ML.NET | Accord.NET |
|:--------|:---------|:-----------|:---------------|:-------|:-----------|
| **Neural Network Architectures** | **100+** | 50+ | 30+ | ~10 | ~15 |
| **Classical ML Algorithms** | **106+** | None | None | ~30 | ~50 |
| **Computer Vision Models** | **50+** | Via PyTorch | Via TF | Limited | Basic |
| **Audio Processing** | **90+** | Limited | Limited | None | Basic |
| **Reinforcement Learning** | **80+ agents** | Manual | Limited | None | None |
| **LoRA/Fine-tuning** | **37+ adapters** | Manual | None | None | None |
| **HuggingFace Integration** | **Native** | Partial | Partial | None | None |
| **Distributed Training** | **DDP/FSDP/ZeRO** | DDP only | MirroredStrategy | None | None |

### Performance Advantages

| Benchmark | AiDotNet | TorchSharp | TensorFlow.NET |
|:----------|:---------|:-----------|:---------------|
| **SIMD Optimizations** | **Native** | Via LibTorch | Via TF Runtime |
| **Memory<T> Support** | **Native** | No | No |
| **Span<T> Operations** | **Full** | Limited | Limited |
| **AOT Compilation** | **Supported** | Limited | No |
| **Startup Time** | **Fast** | Slow (Python runtime) | Slow (TF runtime) |

**Key Performance Features:**
- **SIMD-accelerated tensor operations** - Native AVX2/AVX-512 support
- **BLAS integration** - Optional Intel MKL/OpenBLAS for matrix operations
- **GPU acceleration** - CUDA and OpenCL support without Python dependencies
- **Memory efficient** - Uses Memory<T>/Span<T> for zero-copy operations
- **No Python runtime** - Pure .NET execution, no interop overhead

### Why Not TorchSharp?

TorchSharp wraps PyTorch's C++ runtime (LibTorch), which means:
- **Large runtime dependency** (~700MB+ LibTorch binaries)
- **Slower startup** - Must load PyTorch runtime
- **Limited .NET integration** - Array copying between .NET and LibTorch
- **No classical ML** - Only deep learning, no traditional algorithms
- **Manual everything** - No AutoML, no hyperparameter optimization

AiDotNet provides:
- **Pure .NET implementation** - No external runtime dependencies
- **Instant startup** - No runtime initialization overhead
- **Native .NET types** - Memory<T>, Span<T>, IAsyncEnumerable<T>
- **106+ classical ML algorithms** - Full traditional ML support
- **Built-in AutoML** - Automatic model selection and tuning

### Why Not TensorFlow.NET?

TensorFlow.NET wraps TensorFlow's C runtime, which means:
- **Complex setup** - Requires TensorFlow native libraries
- **Version compatibility issues** - TF version must match wrapper version
- **Limited Keras support** - Incomplete high-level API
- **Resource heavy** - TensorFlow runtime consumes significant memory

AiDotNet provides:
- **Simple NuGet install** - Just `dotnet add package AiDotNet`
- **Always compatible** - No version matching required
- **High-level API** - AiModelBuilder for easy model creation
- **Lightweight** - Only load what you use

### Why Not ML.NET?

ML.NET is Microsoft's official ML library, but:
- **Limited neural networks** - Only basic architectures (~10)
- **No computer vision** - Must use ONNX models
- **No audio processing** - No speech/audio support
- **No reinforcement learning** - No RL agents
- **No HuggingFace** - No transformer model support

AiDotNet provides:
- **100+ neural network architectures** - CNN, RNN, Transformer, GAN, VAE, GNN
- **50+ computer vision models** - YOLO v8-11, DETR, Mask R-CNN, SAM
- **90+ audio models** - Whisper, TTS, music generation
- **80+ RL agents** - DQN, PPO, SAC, multi-agent systems
- **Native HuggingFace** - Load and fine-tune transformer models

### Feature Depth Comparison

#### Neural Networks
| Architecture Type | AiDotNet | Others |
|:------------------|:---------|:-------|
| Convolutional (CNN) | 15+ variants | Basic |
| Recurrent (RNN/LSTM/GRU) | 10+ variants | Basic |
| Transformer | 20+ variants | Manual |
| GAN | 15+ variants | Manual |
| VAE | 10+ variants | Manual |
| Graph Neural Networks | 10+ variants | None/.NET |
| Diffusion Models | 20+ variants | None/.NET |
| NeRF/3D | 5+ variants | None/.NET |

#### Training Capabilities

| Capability | AiDotNet | TorchSharp | ML.NET |
|:-----------|:---------|:-----------|:-------|
| Mixed Precision (FP16/BF16) | **Yes** | Yes | No |
| Gradient Checkpointing | **Yes** | Yes | No |
| Multi-GPU Training | **DDP/FSDP/ZeRO** | DDP | No |
| AutoML | **Built-in** | No | AutoML.NET |
| Hyperparameter Optimization | **Built-in** | No | Limited |
| Meta-Learning | **15+ methods** | No | No |
| Self-Supervised Learning | **10+ methods** | Manual | No |

### Summary: When to Use AiDotNet

**Choose AiDotNet when you need:**
- A single framework that does everything (classical ML + deep learning + RL)
- Native .NET performance without Python/C++ runtime dependencies
- State-of-the-art models (YOLO v11, Whisper, Stable Diffusion)
- HuggingFace model integration
- Distributed training (DDP, FSDP, ZeRO)
- LoRA fine-tuning for LLMs
- Production deployment with AiDotNet.Serving

**Consider alternatives when:**
- You need PyTorch ecosystem compatibility → TorchSharp
- You have existing TensorFlow models → TensorFlow.NET
- You only need basic ML with Microsoft support → ML.NET

## Getting Help

- [Samples Repository](/samples/) - Complete, runnable examples
- [API Reference](/api/) - Full API documentation
- [GitHub Issues](https://github.com/ooples/AiDotNet/issues) - Report bugs
- [GitHub Discussions](https://github.com/ooples/AiDotNet/discussions) - Ask questions

---

## About

AiDotNet is developed and maintained by [Ooples Finance](https://github.com/ooples) with contributions from the community.

Licensed under [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).
