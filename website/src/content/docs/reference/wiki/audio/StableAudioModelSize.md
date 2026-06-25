---
title: "StableAudioModelSize"
description: "Specifies the size variant of the Stable Audio model."
section: "API Reference"
---

`Enums` · `AiDotNet.Audio.StableAudio`

Specifies the size variant of the Stable Audio model.

## For Beginners

Think of model sizes like different quality levels:

- **Small**: Fast generation, good for experimentation (300M parameters)
- **Base**: Balanced quality and speed (800M parameters)
- **Large**: Best quality, requires more resources (1.5B parameters)
- **Open**: Open-source variant with permissive license

Start with Small for testing, use Large for production.

## How It Works

Stable Audio is a latent diffusion model by Stability AI for high-quality audio generation.
It uses a Diffusion Transformer (DiT) architecture instead of U-Net for improved quality
and supports variable-length audio generation.

## Fields

| Field | Summary |
|:-----|:--------|
| `Base` | Base model variant (800M parameters). |
| `Large` | Large model variant (1.5B parameters). |
| `Open` | Stable Audio Open variant. |
| `Small` | Small model variant (300M parameters). |
| `V2` | Stable Audio 2.0 variant. |

