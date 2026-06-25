---
title: "RT2ActionTokenizer<T>"
description: "RT-2's action-as-text tokenizer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Robotics`

RT-2's action-as-text tokenizer.

## How It Works

Implements the RT-2 paper's action representation: each continuous action dimension is
uniformly discretized into `NumBins` bins (default 256) over its range, and each
bin is associated with one of the `NumBins` least frequently used tokens in the
vision-language model's vocabulary. The model emits these tokens autoregressively just
as it would emit ordinary text tokens; downstream decoding maps each token back to a
continuous action value in the centre of its bin.

**References:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RT2ActionTokenizer(Int32,Int32,Int32,Double[],Double[])` | Initialises the tokenizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionDim` | Number of continuous action dimensions emitted per timestep. |
| `NumBins` | Number of discrete bins per action dimension (RT-2 paper: 256). |
| `TokenIdEndExclusive` | One past the last action-bin token ID. |
| `TokenIdOffset` | Vocabulary token-ID where the first action bin lives. |
| `VocabSize` | Total vocabulary size — equals `TokenIdEndExclusive` because the action bins live in the LAST `NumBins` token IDs of the vocab per paper §3.2. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ActionDimOfPosition(Int32)` | Given a position in the emitted action-token stream, returns which continuous action dimension it controls. |
| `DecodeAction(Int32[])` | Decodes vocabulary token IDs (one per action dimension) into a continuous action tensor of length `ActionDim`. |
| `DecodeHorizon(Int32[],Int32)` | Decodes a flat token stream of length `horizon * ActionDim` into a horizon-by-dim action tensor. |
| `EncodeAction(Tensor<>)` | Encodes a continuous action vector into `ActionDim` vocabulary token IDs. |
| `EncodeAction(Vector<>)` | Encodes a per-dimension continuous action vector into vocabulary token IDs. |
| `EncodeHorizon(Tensor<>,Int32)` | Encodes a multi-step horizon action (shape `[horizon, ActionDim]` or `[horizon * ActionDim]`) into a flat token stream of length `horizon * ActionDim`. |
| `GreedyActionToken(Tensor<>)` | Selects the action-bin token with the highest logit (greedy argmax over the tokenizer's vocabulary slice). |
| `IsActionToken(Int32)` | Returns true when the supplied token ID falls in this tokenizer's action-bin range. |

