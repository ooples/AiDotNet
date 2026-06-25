---
title: "MiniCPMo<T>"
description: "MiniCPM-o: omnimodal VLM with speech and real-time streaming support."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

MiniCPM-o: omnimodal VLM with speech and real-time streaming support.

## For Beginners

MiniCPM-o extends MiniCPM-V beyond just images to handle
speech and real-time video streaming. It can see, hear, and speak — processing live video
feeds, understanding spoken questions, and responding with generated speech, all in a
compact model designed for on-device deployment. Think of it as a GPT-4o-level multimodal
assistant that can run on your own hardware, supporting real-time conversations about what
it sees and hears. Default values follow the original paper settings.

## How It Works

MiniCPM-o (OpenBMB, 2025) extends MiniCPM-V to become an omnimodal model that handles vision,
speech, and real-time streaming in a unified architecture. It adds speech understanding and
generation capabilities alongside the existing visual understanding, enabling applications like
live video narration, real-time visual question answering with voice input/output, and
multimodal live streaming analysis. Despite its compact size, it achieves GPT-4o-level
performance on many multimodal benchmarks.

**References:**

- Paper: "MiniCPM-o: A GPT-4o Level MLLM for Vision, Speech and Multimodal Live Streaming" (2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using MiniCPM-o's omni-modal architecture. |

