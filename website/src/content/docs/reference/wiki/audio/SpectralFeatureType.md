---
title: "SpectralFeatureType"
description: "Types of spectral features that can be extracted."
section: "API Reference"
---

`Enums` · `AiDotNet.Audio.Features`

Types of spectral features that can be extracted.

## Fields

| Field | Summary |
|:-----|:--------|
| `All` | All available features. |
| `Bandwidth` | Spectral bandwidth (spread around centroid). |
| `Basic` | All basic features (centroid, bandwidth, rolloff, flux, flatness). |
| `Centroid` | Spectral centroid (center of mass of spectrum). |
| `Contrast` | Spectral contrast (difference between peaks and valleys in sub-bands). |
| `Flatness` | Spectral flatness (noisiness vs tonality). |
| `Flux` | Spectral flux (frame-to-frame spectral change). |
| `None` | No features. |
| `Rolloff` | Spectral rolloff (frequency below which most energy is concentrated). |
| `ZeroCrossingRate` | Zero crossing rate (sign change frequency). |

