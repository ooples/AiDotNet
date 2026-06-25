---
title: "FrequencyAttentionType"
description: "Specifies the type of frequency attention to use in FEDformer."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the type of frequency attention to use in FEDformer.

## For Beginners

These are two different ways to analyze frequency content:

- Fourier is simpler and works well for stationary patterns
- Wavelet can handle patterns that change over time better

## Fields

| Field | Summary |
|:-----|:--------|
| `Fourier` | Uses Fourier transform for frequency attention. |
| `Wavelet` | Uses Wavelet transform for frequency attention. |

