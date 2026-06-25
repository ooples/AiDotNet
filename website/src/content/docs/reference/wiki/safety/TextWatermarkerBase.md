---
title: "TextWatermarkerBase<T>"
description: "Abstract base class for text watermarking modules."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Watermarking`

Abstract base class for text watermarking modules.

## For Beginners

This base class provides common code for all text watermarkers.
Each watermarker type extends this and adds its own way of embedding invisible
signatures in AI-generated text.

## How It Works

Provides shared infrastructure for text watermarkers including watermark strength
configuration and common token processing. Concrete implementations provide
the actual watermarking strategy (sampling, lexical, syntactic).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TextWatermarkerBase(Double)` | Initializes the text watermarker base. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectWatermark(String)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `WatermarkStrength` | The watermark strength factor (0.0 to 1.0). |

