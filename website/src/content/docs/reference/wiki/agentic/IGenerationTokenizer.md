---
title: "IGenerationTokenizer"
description: "The minimal tokenizer contract the local generation engine needs: turn text into token ids, turn token ids back into text, and know which token marks end-of-sequence."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.Models.Local`

The minimal tokenizer contract the local generation engine needs: turn text into token ids, turn token
ids back into text, and know which token marks end-of-sequence.

## For Beginners

Models don't read text directly — they read numbers (token ids). This turns
your prompt into those numbers, turns the model's numbers back into readable text, and tells the engine
the special "stop here" token so it knows when the model is done.

## How It Works

This is intentionally narrower than the full `ITokenizer` so
the engine stays decoupled and trivially testable. `TokenizerGenerationAdapter` bridges a
real repo tokenizer to this seam.

## Properties

| Property | Summary |
|:-----|:--------|
| `EosTokenId` | Gets the id of the end-of-sequence token. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Decode(IReadOnlyList<Int32>)` | Decodes token ids back into text. |
| `Encode(String)` | Encodes text into token ids. |

