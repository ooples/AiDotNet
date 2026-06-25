---
title: "IIncrementalCausalLanguageModel<T>"
description: "An `ICausalLanguageModel` that supports incremental (KV-cached) decoding: it processes the prompt once, caches the per-position state, and then advances one token at a time without recomputing the whole sequence."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.Models.Local`

An `ICausalLanguageModel` that supports incremental (KV-cached) decoding: it processes the
prompt once, caches the per-position state, and then advances one token at a time without recomputing the
whole sequence. This is the fast path for autoregressive generation.

## For Beginners

Without caching, predicting each new word re-reads the entire conversation —
slower and slower as it grows. With caching, the model remembers the work it already did and only looks at
the one new word each time. This interface is how a model advertises "I can do the fast, remember-as-you-go
version"; the engine uses it automatically when available and falls back to the simple way otherwise.

## How It Works

The base `Int32})` re-feeds the full context each step, which
is correct but O(n²) over a generation. A model that maintains a key/value cache implements this interface
so `LocalEngineChatClient` can drive it incrementally: `Int32})` primes
the cache with the prompt and returns the first next-token logits, then each `Int32)`
feeds a single new token and returns the following logits. `ResetCache` clears state between
independent generations.

## Methods

| Method | Summary |
|:-----|:--------|
| `AppendToken(Int32)` | Appends a single token to the cached context and returns the logits for the token after it. |
| `ResetCache` | Clears any cached decoding state, so the next `Int32})` begins fresh. |
| `StartSequence(IReadOnlyList<Int32>)` | Processes the prompt, populating the cache, and returns the logits for the first generated token. |

