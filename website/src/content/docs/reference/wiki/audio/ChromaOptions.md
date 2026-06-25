---
title: "ChromaOptions"
description: "Options for chroma feature extraction."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Features`

Options for chroma feature extraction.

## For Beginners

These options configure the Chroma model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChromaOptions` | Initializes a new instance with default values. |
| `ChromaOptions(ChromaOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Normalize` | Gets or sets whether to L2-normalize each chroma frame. |
| `NumOctaves` | Gets or sets the number of octaves to include. |
| `TuningFrequency` | Gets or sets the tuning frequency for A4. |

