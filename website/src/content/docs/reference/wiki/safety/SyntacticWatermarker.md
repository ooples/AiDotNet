---
title: "SyntacticWatermarker<T>"
description: "Text watermarker that embeds watermarks through syntactic structure rearrangement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Watermarking`

Text watermarker that embeds watermarks through syntactic structure rearrangement.

## For Beginners

This watermarker changes the structure of sentences without
changing their meaning. For example, "The dog bit the man" vs "The man was bitten
by the dog" both mean the same thing but encode different watermark bits.

## How It Works

Encodes watermark bits through syntactic choices: active vs passive voice, clause
ordering, comma placement patterns, and sentence structure variations. Detection
analyzes the statistical distribution of syntactic patterns against expected distributions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SyntacticWatermarker(Double)` | Initializes a new syntactic watermarker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectWatermark(String)` |  |
| `EvaluateText(String)` |  |

