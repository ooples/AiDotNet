---
title: "ITextWatermarker<T>"
description: "Interface for text watermarking modules that embed and detect watermarks in text."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Safety.Watermarking`

Interface for text watermarking modules that embed and detect watermarks in text.

## For Beginners

A text watermarker adds an invisible signature to AI-generated text.
Humans can't see the watermark, but a detector can find it later to prove the text was
AI-generated. This helps with transparency and regulatory compliance.

## How It Works

Text watermarkers modify the token distribution or lexical/syntactic structure of text
to embed an imperceptible watermark that can later be detected to prove AI origin.
Approaches include sampling distribution modification (SynthID-style), synonym
substitution, and structural rearrangement.

**References:**

- SynthID-Text: Production text watermarking at scale (Google DeepMind, Nature 2024)
- SoK: Systematization of watermarking across modalities (2024, arxiv:2411.18479)

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectWatermark(String)` | Detects the watermark confidence score in the given text (0.0 = no watermark, 1.0 = certain). |

