---
title: "RSGPT<T>"
description: "RSGPT: remote sensing GPT based on InstructBLIP architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.RemoteSensing`

RSGPT: remote sensing GPT based on InstructBLIP architecture.

## For Beginners

RSGPT is a vision-language model for remote sensing image
captioning and question answering. Default values follow the original paper settings.

## How It Works

RSGPT (2024) is a remote sensing vision-language model built on the InstructBLIP architecture.
It adapts the Q-Former cross-modal bridge for remote sensing imagery, providing satellite
image captioning, visual question answering, and scene understanding capabilities with
domain-specific visual encoders fine-tuned on remote sensing datasets.

**References:**

- Paper: "RSGPT: A Remote Sensing Vision Language Model and Benchmark (Various, 2024)"

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from a remote sensing image using RSGPT's InstructBLIP pipeline. |

