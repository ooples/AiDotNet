---
title: "AudioLDMModelSize"
description: "Specifies the size variant of the AudioLDM model."
section: "API Reference"
---

`Enums` · `AiDotNet.Audio.AudioLDM`

Specifies the size variant of the AudioLDM model.

## For Beginners

Think of model sizes like different quality levels:

- **Small**: Fast generation, good for experimentation (345M parameters)
- **Base**: Balanced quality and speed (740M parameters)
- **Large**: Best quality, requires more resources (1.5B parameters)

Start with Small for testing, use Large for final production.

## How It Works

AudioLDM (Audio Latent Diffusion Model) comes in different sizes balancing quality
and computational requirements. All variants use latent diffusion in a compressed
audio representation space.

## Fields

| Field | Summary |
|:-----|:--------|
| `Base` | Base model variant (740M parameters). |
| `Large` | Large model variant (1.5B parameters). |
| `Music` | Music-specialized variant. |
| `Small` | Small model variant (345M parameters). |
| `V2` | AudioLDM-2 variant with improved architecture. |

