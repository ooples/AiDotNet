---
title: "ChromaprintOptions"
description: "Configuration options for Chromaprint fingerprinting."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Fingerprinting`

Configuration options for Chromaprint fingerprinting.

## For Beginners

These options configure the Chromaprint model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChromaprintOptions` | Initializes a new instance with default values. |
| `ChromaprintOptions(ChromaprintOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextSize` | Gets or sets the number of context frames for hashing. |
| `FftSize` | Gets or sets the FFT size. |
| `HashStep` | Gets or sets the step between hash computations. |
| `HopLength` | Gets or sets the hop length. |
| `MaxBitDifference` | Gets or sets the maximum bit difference for approximate matching. |
| `SampleRate` | Gets or sets the sample rate (default 11025 Hz for efficiency). |

