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

var result = await new PredictionModelBuilder<double, double[], double>()
    .ConfigureModel(new NeuralNetwork<double>(inputSize: 4, hiddenSize: 16, outputSize: 3))
    .ConfigureOptimizer(new AdamOptimizer<double>())
    .ConfigurePreprocessing()
    .BuildAsync(features, labels);

var prediction = result.Model.Predict(newSample);
```

## What do you want to build?

<div class="grid-container">

| Task | Documentation |
|:-----|:--------------|
| **Classify data** | [Classification Tutorial](/tutorials/classification/) |
| **Predict values** | [Regression Tutorial](/tutorials/regression/) |
| **Detect objects** | [Computer Vision Tutorial](/tutorials/computer-vision/) |
| **Process speech** | [Audio Tutorial](/tutorials/audio/) |
| **Build a chatbot** | [RAG Tutorial](/tutorials/rag/) |
| **Fine-tune LLMs** | [LoRA Tutorial](/tutorials/lora/) |
| **Scale training** | [Distributed Training Tutorial](/tutorials/distributed/) |

</div>

## Why AiDotNet?

### Compared to ML.NET

| Feature | AiDotNet | ML.NET |
|:--------|:---------|:-------|
| Neural Network Architectures | **100+** | ~10 |
| Computer Vision | **50+ models** | Limited |
| Audio Processing | **90+ models** | None |
| Reinforcement Learning | **80+ agents** | None |
| HuggingFace Integration | **Full** | None |

### Compared to Accord.NET

| Feature | AiDotNet | Accord.NET |
|:--------|:---------|:-----------|
| Active Development | **Yes** | Minimal |
| Modern .NET Support | **.NET 8+** | Limited |
| GPU Acceleration | **CUDA, OpenCL** | None |
| Deep Learning | **Full stack** | Basic |

## Getting Help

- [Samples Repository](/samples/) - Complete, runnable examples
- [API Reference](/api/) - Full API documentation
- [GitHub Issues](https://github.com/ooples/AiDotNet/issues) - Report bugs
- [GitHub Discussions](https://github.com/ooples/AiDotNet/discussions) - Ask questions

---

## About

AiDotNet is developed and maintained by [Ooples Finance](https://github.com/ooples) with contributions from the community.

Licensed under [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).
