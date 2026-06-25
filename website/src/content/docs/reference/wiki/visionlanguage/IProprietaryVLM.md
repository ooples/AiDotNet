---
title: "IProprietaryVLM<T>"
description: "Interface for reference implementations of proprietary VLM architectures."
section: "API Reference"
---

`Interfaces` · `AiDotNet.VisionLanguage.Interfaces`

Interface for reference implementations of proprietary VLM architectures.

## How It Works

Proprietary VLMs represent state-of-the-art commercial models from major AI labs.
These reference implementations approximate their published architectures for
understanding the design space and benchmarking open alternatives.

## Properties

| Property | Summary |
|:-----|:--------|
| `LanguageModelName` | Gets the name of the language model backbone. |
| `Provider` | Gets the name of the proprietary model provider. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Chat(Tensor<>,String)` | Generates output from an image with a text prompt in chat-style interaction. |

