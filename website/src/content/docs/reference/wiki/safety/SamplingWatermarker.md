---
title: "SamplingWatermarker<T>"
description: "Text watermarker that modifies token sampling distributions to embed watermarks (SynthID-style)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Watermarking`

Text watermarker that modifies token sampling distributions to embed watermarks (SynthID-style).

## For Beginners

This watermarker subtly biases which words the AI chooses.
It creates a list of "preferred" words for each position based on a secret key.
The bias is invisible to readers, but a detector can measure whether the text
uses more "preferred" words than expected by chance.

## How It Works

Uses a hash-based green/red list partition of the vocabulary conditioned on previous tokens.
Tokens in the "green list" are slightly favored during generation. Detection measures the
statistical over-representation of green-list tokens using a z-score test.

**References:**

- SynthID-Text: Production text watermarking at scale (Google DeepMind, Nature 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SamplingWatermarker(Double,Int32,Double)` | Initializes a new sampling watermarker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectWatermark(String)` |  |
| `EvaluateText(String)` |  |

