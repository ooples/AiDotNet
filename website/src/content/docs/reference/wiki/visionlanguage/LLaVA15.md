---
title: "LLaVA15<T>"
description: "LLaVA-1.5: improved baselines with visual instruction tuning using MLP cross-modal connector."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

LLaVA-1.5: improved baselines with visual instruction tuning using MLP cross-modal connector.

## For Beginners

LLaVA-1.5 is an improved version of the original LLaVA (Large
Language and Vision Assistant) that makes simple but effective upgrades: it replaces the
single linear projection with a two-layer MLP connector (for better visual-to-text mapping),
uses a higher-resolution CLIP-ViT-L/14 encoder at 336 pixels (for sharper visual detail),
and adds academic VQA training data. Despite these minimal changes, it achieved
state-of-the-art results on 11 benchmarks, demonstrating that careful engineering of
the basics often matters more than complex architectural innovations. Default values
follow the original paper settings.

## How It Works

LLaVA-1.5 (Liu et al., 2024) improves upon LLaVA by replacing the linear projection with
a two-layer MLP cross-modal connector, using higher-resolution CLIP-ViT-L/14 at 336px,
and adding academic-task-oriented VQA data. It achieves SOTA results on 11 benchmarks with
simple modifications to the original LLaVA architecture.

**References:**

- Paper: "Improved Baselines with Visual Instruction Tuning" (Liu et al., 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using LLaVA-1.5's MLP cross-modal connector architecture. |

