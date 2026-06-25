---
title: "WatermarkDetector<T>"
description: "Unified watermark detector that combines multiple watermark detection strategies."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Watermarking`

Unified watermark detector that combines multiple watermark detection strategies.

## For Beginners

This detector checks for all known types of text watermarks
at once. It combines multiple detection methods to be more accurate than any single
method alone.

## How It Works

Runs sampling, lexical, and syntactic watermark detectors in parallel and aggregates
their scores to determine whether text contains any type of watermark.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WatermarkDetector` | Initializes a new composite watermark detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectWatermark(String)` |  |
| `EvaluateText(String)` |  |

