---
title: "MiniCPMV<T>"
description: "MiniCPM-V: efficient on-device VLM with strong OCR and multilingual capabilities."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

MiniCPM-V: efficient on-device VLM with strong OCR and multilingual capabilities.

## For Beginners

MiniCPM-V is designed to bring GPT-4V-level vision understanding
to your phone. Despite being small enough to run on mobile devices, it achieves surprisingly
strong performance on visual tasks like reading text in images (OCR), understanding documents,
and answering questions about photos. It supports over 30 languages and uses adaptive visual
encoding to efficiently handle images at different resolutions. This makes it ideal for
applications where you need on-device visual AI without cloud connectivity. Default values
follow the original paper settings.

## How It Works

MiniCPM-V (OpenBMB, 2024) is an efficient multimodal model designed to run on mobile devices
while achieving performance comparable to GPT-4V on many tasks. Despite its small size, it
features strong OCR capabilities for reading text in images and supports over 30 languages.
It uses adaptive visual encoding to handle different image resolutions efficiently and a
compact language backbone optimized for on-device deployment.

**References:**

- Paper: "MiniCPM-V: A GPT-4V Level MLLM on Your Phone" (OpenBMB, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using MiniCPM-V's slice-then-compress architecture. |

