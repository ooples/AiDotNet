---
title: "MambaCausalLanguageModel<T>"
description: "A KV-cached `IIncrementalCausalLanguageModel` adapter over `MambaLanguageModel`."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Local`

A KV-cached `IIncrementalCausalLanguageModel` adapter over `MambaLanguageModel`.
It drives the model's per-token `MambaModelState{` fast path, carrying the recurrent
state (causal-conv window + selective-scan hidden state) so each new token costs O(1) instead of
reprocessing the whole context.

## For Beginners

A plain forward pass re-reads the entire conversation every time it predicts a
word, which gets slower as the text grows. Mamba can instead remember a small summary of everything so far
and update it with just the new word — same answer, far less work. This adapter exposes that
"remember-as-you-go" ability to the chat engine.

## How It Works

The incremental path is mathematically equivalent to the full-sequence forward (Gu & Dao 2023): feeding
tokens one at a time while carrying state reproduces `Predict`'s logits at
every position. This is verified by unit tests that assert step-by-step output matches the parallel scan.
The base `Int32})` still runs a full forward (used when a caller does not opt into the
cache); the fast path is `Int32})` + `Int32)`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MambaCausalLanguageModel(MambaLanguageModel<>,Int32)` | Initializes a new KV-cached adapter over a Mamba language model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `VocabularySize` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AppendToken(Int32)` |  |
| `NextTokenLogits(IReadOnlyList<Int32>)` |  |
| `ResetCache` |  |
| `StartSequence(IReadOnlyList<Int32>)` |  |

