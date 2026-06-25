---
title: "AudioSealWatermarker<T>"
description: "Audio watermarker using AudioSeal-style localized watermarking for voice cloning detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Watermarking`

Audio watermarker using AudioSeal-style localized watermarking for voice cloning detection.

## For Beginners

This watermarker adds tiny signatures to small segments of
audio. Unlike a global watermark, it can tell you exactly which parts of a recording
are AI-generated and which parts are real — useful for detecting partial deepfakes.

## How It Works

Implements a localized watermarking approach inspired by Meta AI's AudioSeal.
Instead of embedding a single global watermark, embeds localized watermarks
that can identify which specific segments are AI-generated even in partially
modified audio.

**References:**

- AudioSeal: Localized watermarking for speech (Meta AI, 2024, arxiv:2401.17264)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioSealWatermarker(Double,Int32)` | Initializes a new AudioSeal-style watermarker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectWatermark(Vector<>,Int32)` |  |
| `EvaluateAudio(Vector<>,Int32)` |  |

