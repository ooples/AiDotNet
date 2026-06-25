---
title: "TokenSampler<T>"
description: "Selects the next token id from a logits vector according to `LocalSamplingOptions`: greedy (argmax) when temperature is zero, otherwise temperature-scaled softmax sampling restricted by optional top-k and top-p filters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Local`

Selects the next token id from a logits vector according to `LocalSamplingOptions`:
greedy (argmax) when temperature is zero, otherwise temperature-scaled softmax sampling restricted by
optional top-k and top-p filters.

## For Beginners

This is the "dice roll" step. With temperature 0 it isn't a roll at all — it
just takes the single most likely token. Otherwise it turns the scores into probabilities, optionally
throws away the unlikely options (top-k / top-p), and then picks one at random in proportion to how
likely each is.

## How It Works

The sampler owns a single `Random` (seeded from `Seed` when
provided), so a fixed seed yields a reproducible token stream. Logits are read through
`Object)`, so the same code path serves `Single` and
`Double` models.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TokenSampler(Nullable<Int32>)` | Initializes a new sampler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Sample(Vector<>,LocalSamplingOptions,IReadOnlyCollection<Int32>)` | Chooses the next token id from the supplied logits, optionally restricted to an allowed set (constrained decoding). |

