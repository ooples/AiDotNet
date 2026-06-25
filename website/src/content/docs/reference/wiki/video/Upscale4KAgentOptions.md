---
title: "Upscale4KAgentOptions"
description: "Configuration options for the Upscale4KAgent agentic multi-model pipeline."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the Upscale4KAgent agentic multi-model pipeline.

## For Beginners

Instead of using one model for everything, Upscale4KAgent acts
like a "manager" that looks at each frame and decides which combination of upscaling
models will produce the best result. It can chain multiple models together and check
quality at each step, similar to how a human editor would approach video upscaling.

## How It Works

Upscale4KAgent (2025) orchestrates multiple specialized SR models in an agentic pipeline:

- Quality assessment agent: evaluates each frame to determine optimal processing strategy
- Multi-model routing: dynamically selects and chains SR models based on content analysis
- Iterative refinement: applies progressive upscaling stages with quality checkpoints
- Resolution-adaptive: handles arbitrary input resolution up to 4K output

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Upscale4KAgentOptions` | Initializes a new instance with default values. |
| `Upscale4KAgentOptions(Upscale4KAgentOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxAgentSteps` | Gets or sets the maximum number of agent decision steps per frame. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels in the backbone. |
| `NumResBlocks` | Gets or sets the number of residual blocks per stage backbone. |
| `NumStages` | Gets or sets the number of pipeline stages for iterative refinement. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `QualityThreshold` | Gets or sets the quality threshold (0.0-1.0) for the agent to accept a result. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `Variant` | Gets or sets the model variant. |

