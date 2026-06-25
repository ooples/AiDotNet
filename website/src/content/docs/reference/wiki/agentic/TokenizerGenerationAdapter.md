---
title: "TokenizerGenerationAdapter"
description: "Bridges a full repo `ITokenizer` to the engine's minimal `IGenerationTokenizer` seam, so any AiDotNet tokenizer can drive `LocalEngineChatClient`."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Local`

Bridges a full repo `ITokenizer` to the engine's minimal `IGenerationTokenizer`
seam, so any AiDotNet tokenizer can drive `LocalEngineChatClient`.

## For Beginners

The library's tokenizers do a lot more than generation needs. This adapter
exposes just the three things the generation loop uses (encode, decode, end-of-sequence id), so you can
plug a real tokenizer into the local engine without it depending on the larger tokenizer surface.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TokenizerGenerationAdapter(ITokenizer)` | Initializes a new adapter over the given tokenizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EosTokenId` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Decode(IReadOnlyList<Int32>)` |  |
| `Encode(String)` |  |

