---
title: "GenerationRequest<T>"
description: "Represents a request for text generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Serving.ContinuousBatching`

Represents a request for text generation.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxNewTokens` | Maximum number of new tokens to generate. |
| `NumBeams` | Number of beams for beam search. |
| `OnTokenGenerated` | Callback for streaming tokens. |
| `Priority` | Priority for scheduling (higher = more important). |
| `PromptTokenIds` | Token IDs of the prompt. |
| `RepetitionPenalty` | Repetition penalty (1.0 = no penalty). |
| `StopTokenIds` | Additional stop token IDs. |
| `Temperature` | Temperature for sampling (higher = more random). |
| `TopK` | Top-k sampling (0 = disabled). |
| `TopP` | Top-p (nucleus) sampling threshold. |
| `UseBeamSearch` | Whether to use beam search. |
| `UserContext` | Optional user context. |

