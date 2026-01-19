---
layout: default
title: Changelog
parent: Community
nav_order: 2
permalink: /community/changelog/
---

# Changelog
{: .no_toc }

All notable changes to AiDotNet are documented here.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## [Unreleased]

### Added
- Comprehensive documentation site with tutorials
- 30+ sample applications covering all feature categories
- End-to-end applications: ChatbotWithRAG, SpeechAssistant, ImageClassificationWebApp
- Expanded "Why AiDotNet" comparison against TorchSharp, TensorFlow.NET, ML.NET
- Navigation configuration for GitHub Pages
- Community documentation (contributing guide, roadmap)

### Changed
- Migrated Vector/Matrix/Tensor data from T[] to Memory<T> for better performance
- Updated GPU training infrastructure for LSTM/GRU, GNN, and activations

### Fixed
- Diffusion conv GPU training with auto eigenbasis
- Applied dotnet format across codebase

---

## [0.x.x] - Recent Releases

### GPU and Performance
- **SIMD and BLAS optimizations** for CPU tensor operations
- **GPU training infrastructure** for LSTM/GRU, GNN, and activation functions
- **Memory<T> migration** for zero-copy tensor operations
- **GPU kernel optimizations** for common operations

### Neural Networks
- 100+ neural network architectures including:
  - Transformer variants (BERT, GPT, ViT, CLIP)
  - Diffusion models (Stable Diffusion components)
  - Graph Neural Networks (GCN, GAT, GraphSAGE)
  - 3D and NeRF models

### Classical ML
- 106+ classical machine learning algorithms
- 28 classification algorithms
- 41 regression algorithms
- 20+ clustering algorithms

### Computer Vision
- 50+ computer vision models
- YOLO v8-11 object detection
- DETR and Faster R-CNN
- Mask R-CNN instance segmentation
- SAM (Segment Anything Model)
- OCR models

### Audio Processing
- 90+ audio processing models
- Whisper speech recognition
- Text-to-Speech synthesis
- Audio classification
- Music generation

### Reinforcement Learning
- 80+ RL agents
- DQN, Double DQN, Dueling DQN, Rainbow
- PPO, A2C, TRPO
- SAC, DDPG, TD3
- Multi-agent systems (MADDPG, QMIX, MAPPO)

### RAG Components
- 50+ RAG components
- Sentence transformer embeddings
- In-memory and FAISS vector stores
- Dense, sparse, and hybrid retrievers
- Cross-encoder rerankers

### LoRA Fine-tuning
- 37+ LoRA adapters
- QLoRA (4-bit quantized)
- DoRA (Weight-Decomposed)
- AdaLoRA (Adaptive)
- VeRA, LoKr, LoHa variants

### Distributed Training
- 10+ distributed strategies
- DDP (Distributed Data Parallel)
- FSDP (Fully Sharded Data Parallel)
- ZeRO optimization (Stage 1/2/3)
- Pipeline and Tensor parallelism

### Meta-Learning
- 15+ meta-learning methods
- MAML, Reptile
- Prototypical Networks
- iMAML

### Self-Supervised Learning
- 10+ SSL methods
- SimCLR, MoCo
- DINO, MAE
- BYOL, Barlow Twins

---

## Version History

The detailed version history is available on GitHub:
[https://github.com/ooples/AiDotNet/releases](https://github.com/ooples/AiDotNet/releases)

---

## Versioning

AiDotNet follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward-compatible
- **PATCH**: Bug fixes, backward-compatible

---

## Upgrade Guide

When upgrading to a new major version, check the release notes for:
- Breaking changes and migration paths
- Deprecated features to update
- New features to take advantage of
