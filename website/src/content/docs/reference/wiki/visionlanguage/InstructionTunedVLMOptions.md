---
title: "InstructionTunedVLMOptions"
description: "Base configuration options for instruction-tuned vision-language models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.InstructionTuned`

Base configuration options for instruction-tuned vision-language models.

## For Beginners

These options configure the InstructionTuned model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InstructionTunedVLMOptions` | Initializes a new instance with default values. |
| `InstructionTunedVLMOptions(InstructionTunedVLMOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InstructionArchitectureType` | Gets or sets the instruction-tuned architecture type. |
| `LanguageModelName` | Gets or sets the language model backbone name. |
| `MaxVisualTokens` | Gets or sets the maximum number of visual tokens per image. |
| `ProjectionDim` | Gets or sets the MLP projection hidden dimension (for MLP connector models). |
| `SystemPrompt` | Gets or sets the system prompt for chat mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ValidateVisualSizing` | Fail-fast validation of the size-related options before any layer is allocated. |

