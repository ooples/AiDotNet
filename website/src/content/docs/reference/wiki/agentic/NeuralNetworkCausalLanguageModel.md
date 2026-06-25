---
title: "NeuralNetworkCausalLanguageModel<T>"
description: "Adapts a trained AiDotNet `NeuralNetworkBase` language model (e.g., `MambaLanguageModel`, `GLALanguageModel`, or a Transformer LM head) to the `ICausalLanguageModel` seam, so `LocalEngineChatClient` can run real, fully in-process generation…"
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Local`

Adapts a trained AiDotNet `NeuralNetworkBase` language model (e.g.,
`MambaLanguageModel`, `GLALanguageModel`, or a Transformer LM head) to the
`ICausalLanguageModel` seam, so `LocalEngineChatClient` can run real,
fully in-process generation over the library's own networks.

## For Beginners

This is the bridge that lets the local chat engine talk to a real AiDotNet
network. It turns the running list of tokens into the exact tensor shape the network expects, asks the
network for its prediction, and hands back the scores for the next token — which the engine then samples
from. The result: a chatbot powered entirely by an in-house model, no external service.

## How It Works

AiDotNet's language models accept a one-hot input tensor of shape `[1, sequence, vocab]` and return
logits of the same leading shape. This adapter encodes the context as that one-hot tensor, runs a forward
pass, and returns the final position's logits. It re-feeds the full context each step (no KV-cache yet)
and calls `ResetState` before each pass so recurrent models (Mamba/GLA)
start fresh — correct, if not yet optimal. A KV-cached fast path is a planned follow-up.

**Quantization** composes through this adapter rather than being re-implemented here: quantize the
network with the repository's `ModelCompression` stack first, then wrap the quantized
`NeuralNetworkBase` — the adapter accepts any such network unchanged, so a smaller/faster
model needs no engine-side code.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralNetworkCausalLanguageModel(NeuralNetworkBase<>,Int32,Nullable<Int32>)` | Initializes a new adapter over a network language model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `VocabularySize` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `NextTokenLogits(IReadOnlyList<Int32>)` |  |

