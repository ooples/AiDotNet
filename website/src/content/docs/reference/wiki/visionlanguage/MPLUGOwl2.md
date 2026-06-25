---
title: "MPLUGOwl2<T>"
description: "mPLUG-Owl2: improved modular design for multi-image understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

mPLUG-Owl2: improved modular design for multi-image understanding.

## For Beginners

mPLUG-Owl2 improves on its predecessor with an enhanced visual
abstractor and LLaMA-2 backbone for better multi-image understanding. The key upgrade is
"modality collaboration" — the model learns to better coordinate between different types
of input (images, text, and their relationships) rather than treating them independently.
This gives it improved reasoning capabilities, especially when dealing with multiple
images or complex visual scenes that require understanding spatial relationships and
interactions between objects. Default values follow the original paper settings.

## How It Works

mPLUG-Owl2 (Alibaba, 2024) improves upon mPLUG-Owl with an enhanced visual abstractor
module and LLaMA-2 backbone for better multi-image understanding and reasoning capabilities.

**References:**

- Paper: "mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration" (2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using mPLUG-Owl2's enhanced modular architecture. |
| `GetExtraTrainableLayers` |  |

