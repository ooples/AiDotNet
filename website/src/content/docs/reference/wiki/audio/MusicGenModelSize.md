---
title: "MusicGenModelSize"
description: "Specifies the size variant of the MusicGen model."
section: "API Reference"
---

`Enums` · `AiDotNet.Audio.MusicGen`

Specifies the size variant of the MusicGen model.

## For Beginners

Think of model sizes like different quality levels:

- **Small**: Fast generation, good for experimentation (300M parameters)
- **Medium**: Balanced quality and speed (1.5B parameters)
- **Large**: Best quality, requires more resources (3.3B parameters)

Start with Small for testing, use Large for final production.

## How It Works

MusicGen comes in different sizes balancing quality and computational requirements.
Larger models produce higher quality music but require more memory and compute.

## Fields

| Field | Summary |
|:-----|:--------|
| `Large` | Large model variant (3.3B parameters). |
| `Medium` | Medium model variant (1.5B parameters). |
| `Melody` | Melody model variant (1.5B parameters). |
| `Small` | Small model variant (300M parameters). |
| `Stereo` | Stereo model variant (1.5B parameters). |

