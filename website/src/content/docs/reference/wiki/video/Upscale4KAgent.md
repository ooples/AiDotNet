---
title: "Upscale4KAgent<T>"
description: "Upscale4KAgent: agentic multi-model pipeline for any-resolution to 4K upscaling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

Upscale4KAgent: agentic multi-model pipeline for any-resolution to 4K upscaling.

## For Beginners

Instead of using one model for everything, Upscale4KAgent acts
like a "manager" that looks at each frame and decides which combination of upscaling
models will produce the best result. It can chain multiple models together and check
quality at each step, similar to how a human editor would approach video upscaling.

**Usage:**

## How It Works

Upscale4KAgent (2025) orchestrates multiple specialized SR models in an agentic pipeline:

- Quality assessment agent: evaluates each frame to determine optimal processing strategy
- Multi-model routing: dynamically selects and chains SR models (e.g., Real-ESRGAN for

textures, SwinIR for faces) based on content analysis of each frame region

- Iterative refinement: applies progressive upscaling stages with quality checkpoints,

continuing until QualityThreshold is met or MaxAgentSteps is reached

- Resolution-adaptive: handles arbitrary input resolution with automatic tiling and

overlap for seamless 4K output

**Reference:** "Upscale4KAgent: Agentic Multi-Model Pipeline for 4K Video Upscaling" (2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Upscale4KAgent(NeuralNetworkArchitecture<>,String,Upscale4KAgentOptions)` | Creates an Upscale4KAgent model in ONNX inference mode. |
| `Upscale4KAgent(NeuralNetworkArchitecture<>,Upscale4KAgentOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an Upscale4KAgent model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

