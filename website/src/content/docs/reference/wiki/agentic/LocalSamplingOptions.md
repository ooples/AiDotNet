---
title: "LocalSamplingOptions"
description: "Controls how the next token is chosen from the model's logits: temperature, top-k, top-p (nucleus), and an optional seed for reproducibility."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Agentic.Models.Local`

Controls how the next token is chosen from the model's logits: temperature, top-k, top-p (nucleus), and
an optional seed for reproducibility.

## For Beginners

After the model says how likely each next word-piece is, these settings decide
how to pick one. Low temperature = safe and repetitive; higher = more creative. Top-k ("only consider the
best k options") and top-p ("only consider the most likely options that together cover p% of the
probability") keep the choice from wandering into unlikely tokens.

## How It Works

All values are nullable with sensible behavior when unset. A temperature of `0` (or less) selects
greedy decoding (always the highest-scoring token); higher temperatures increase randomness. Top-k and
top-p restrict sampling to the most likely tokens. These mirror the knobs exposed on
`ChatOptions`, which override these per request.

## Properties

| Property | Summary |
|:-----|:--------|
| `Seed` | Gets or sets the random seed for reproducible sampling. |
| `Temperature` | Gets or sets the sampling temperature. |
| `TopK` | Gets or sets the number of highest-probability tokens to consider. |
| `TopP` | Gets or sets the nucleus-sampling probability mass (0–1): only the most likely tokens whose cumulative probability reaches this value are considered. |

