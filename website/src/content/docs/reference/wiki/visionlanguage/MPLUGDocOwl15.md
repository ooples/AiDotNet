---
title: "MPLUGDocOwl15<T>"
description: "mPLUG-DocOwl 1.5: unified structure learning achieving SOTA on 10 document benchmarks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Document`

mPLUG-DocOwl 1.5: unified structure learning achieving SOTA on 10 document benchmarks.

## For Beginners

mPLUG-DocOwl 1.5 is an improved document understanding model
with unified structure learning for OCR-free processing. Default values follow the original
paper settings.

## How It Works

mPLUG-DocOwl 1.5 (Alibaba, 2024) achieves state-of-the-art on 10 document understanding
benchmarks through unified structure learning. It learns document layout structure alongside
text content in an OCR-free manner, using structure-aware pre-training objectives that teach
the model to understand tables, lists, headers, and hierarchical document organization.

**References:**

- Paper: "mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding" (Alibaba, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from a document image using mPLUG-DocOwl 1.5's unified structure learning. |

