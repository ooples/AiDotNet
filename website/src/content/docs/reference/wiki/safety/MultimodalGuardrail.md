---
title: "MultimodalGuardrail<T>"
description: "Unified guardrail for vision-language models (VLMs) and multimodal AI systems that validates both text and image content together."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Multimodal`

Unified guardrail for vision-language models (VLMs) and multimodal AI systems that
validates both text and image content together.

## For Beginners

When AI systems accept both text and images (like "describe this image"),
attackers can exploit the gap between modalities. An image might be harmless by itself, and a
prompt might be harmless by itself, but together they could trick the AI. This guardrail
checks the combination to catch such attacks.

## How It Works

Provides input/output guardrailing for multimodal systems. For text-only content,
delegates to configured text safety modules. For image-only content, delegates to
configured image safety modules. For combined text+image content, additionally checks
for cross-modal attacks where individually safe content becomes harmful when combined.

**References:**

- Visual prompt injection attacks on GPT-4V (2024)
- MM-SafetyBench: Multimodal safety benchmark (2024)
- Cross-modal jailbreak attacks on multimodal LLMs (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultimodalGuardrail(IReadOnlyList<ITextSafetyModule<>>,IReadOnlyList<IImageSafetyModule<>>,Double)` | Initializes a new multimodal guardrail. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateTextAudio(String,Vector<>,Int32)` |  |
| `EvaluateTextImage(String,Tensor<>)` |  |

