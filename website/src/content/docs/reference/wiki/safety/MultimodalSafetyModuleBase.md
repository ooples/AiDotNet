---
title: "MultimodalSafetyModuleBase<T>"
description: "Abstract base class for multimodal safety modules that analyze cross-modal content."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Multimodal`

Abstract base class for multimodal safety modules that analyze cross-modal content.

## For Beginners

This base class provides common code for modules that check
the combination of different content types (text + image, text + audio) for safety risks.

## How It Works

Provides shared infrastructure for multimodal safety modules including cross-modal
feature extraction and alignment utilities. Concrete implementations provide
the actual cross-modal analysis (text-image alignment, cross-modal consistency, guardrail).

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Vector<>)` |  |
| `EvaluateTextAudio(String,Vector<>,Int32)` |  |
| `EvaluateTextImage(String,Tensor<>)` |  |

