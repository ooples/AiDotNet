---
layout: default
title: Roadmap
parent: Community
nav_order: 3
permalink: /community/roadmap/
---

# Roadmap
{: .no_toc }

Planned features and development priorities for AiDotNet.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Current Focus

### Q1 2026

**Documentation & Developer Experience**
- [x] Comprehensive tutorials for all feature categories
- [x] 30+ sample applications
- [x] End-to-end application examples
- [x] GitHub Pages documentation site
- [ ] Interactive API documentation
- [ ] Video tutorials

**Performance**
- [x] SIMD optimizations for tensor operations
- [x] Memory<T> migration for zero-copy
- [ ] Additional GPU kernel optimizations
- [ ] Intel MKL integration
- [ ] ARM NEON optimizations

---

## Short-Term (Q2 2026)

### HuggingFace Integration
- [ ] Direct model loading from HuggingFace Hub
- [ ] Tokenizer support for popular models
- [ ] Model card parsing
- [ ] Automatic weight conversion

### Training Improvements
- [ ] Mixed precision training (FP16/BF16)
- [ ] Gradient accumulation optimizations
- [ ] Memory-efficient attention
- [ ] Flash Attention support

### New Models
- [ ] LLaMA 3 support
- [ ] Mistral/Mixtral support
- [ ] Phi-3 models
- [ ] Claude tokenizer compatibility

---

## Medium-Term (Q3-Q4 2026)

### Distributed Training
- [ ] Multi-node training improvements
- [ ] Elastic training (fault tolerance)
- [ ] Improved checkpointing
- [ ] Cloud provider integrations (Azure ML, AWS SageMaker)

### Computer Vision
- [ ] YOLO v12 (when released)
- [ ] Grounding DINO
- [ ] Open vocabulary detection
- [ ] Video understanding models

### Audio
- [ ] Whisper large-v3 turbo
- [ ] Real-time transcription
- [ ] Speaker diarization
- [ ] Music understanding models

### LLM Features
- [ ] Speculative decoding
- [ ] KV cache optimizations
- [ ] Continuous batching
- [ ] PagedAttention

---

## Long-Term (2027+)

### Multi-Modal
- [ ] Vision-Language models
- [ ] Audio-Language models
- [ ] Multi-modal embeddings
- [ ] Multi-modal RAG

### Edge Deployment
- [ ] WASM support
- [ ] Mobile optimization (iOS, Android)
- [ ] IoT deployment
- [ ] TinyML support

### Research Features
- [ ] Neural Architecture Search improvements
- [ ] AutoML v2 with more algorithms
- [ ] Federated learning
- [ ] Privacy-preserving ML

### Ecosystem
- [ ] Visual Studio extension
- [ ] Jupyter kernel
- [ ] MLOps integrations
- [ ] Model registry

---

## Feature Requests

Want to see a feature on the roadmap? Here's how:

1. **Check existing issues**: Search [GitHub Issues](https://github.com/ooples/AiDotNet/issues) first
2. **Open a feature request**: Use the feature request template
3. **Discuss in GitHub Discussions**: For broader ideas
4. **Vote on existing requests**: Add a thumbs up to prioritize

---

## How Priorities Are Set

Features are prioritized based on:

1. **Community demand** - Number of requests and votes
2. **Impact** - How many users benefit
3. **Alignment** - Fits AiDotNet's mission
4. **Feasibility** - Technical complexity
5. **Contributor interest** - Available resources

---

## Contributing to the Roadmap

Want to help implement a roadmap item?

1. Comment on the related GitHub issue
2. Discuss your approach
3. Submit a PR when ready

See the [Contributing Guide](/community/contributing/) for details.

---

## Completed Milestones

### 2025
- Initial release with 100+ neural networks
- GPU acceleration via CUDA and OpenCL
- 106+ classical ML algorithms
- Computer vision models (YOLO, DETR)
- Audio processing (Whisper, TTS)
- RAG components
- LoRA fine-tuning
- Distributed training (DDP, FSDP, ZeRO)

### Early 2026
- Comprehensive documentation site
- 30+ sample applications
- End-to-end examples
- Performance optimizations (SIMD, Memory<T>)
