---
title: "MiniGPTv2<T>"
description: "MiniGPT-v2: unified interface for multi-task VL learning with task-specific tokens."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

MiniGPT-v2: unified interface for multi-task VL learning with task-specific tokens.

## For Beginners

MiniGPT-v2 upgrades MiniGPT-4 to handle multiple visual tasks
in a single model using a clever trick: task-specific identifier tokens. Instead of needing
separate models for different tasks, you prepend a special token that tells the model what
type of task you want — like "[vqa]" for visual question answering, "[caption]" for image
captioning, or "[grounding]" for locating objects. The model learns to produce the right
kind of output based on which task token it sees. It also upgrades the language backbone
from Vicuna to LLaMA-2. Default values follow the original paper settings.

## How It Works

MiniGPT-v2 (Chen et al., 2023) upgrades MiniGPT-4 with LLaMA-2 backbone and a unified
multi-task learning framework using task-specific identifier tokens. It supports visual
question answering, image captioning, visual grounding, and referring expression comprehension
through a single model with task-prefixed instructions.

**References:**

- Paper: "MiniGPT-v2: Large Language Model as a Unified Interface for Vision-Language Multi-task Learning" (Chen et al., 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using MiniGPT-v2's unified multi-task Q-Former architecture. |
| `GetExtraTrainableLayers` |  |

