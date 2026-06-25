---
title: "PaddingMode"
description: "Padding mode for STFT centering."
section: "API Reference"
---

`Enums` · `AiDotNet.Diffusion.Audio`

Padding mode for STFT centering.

## Fields

| Field | Summary |
|:-----|:--------|
| `Reflect` | Reflect padding: abcde -> edcba\|abcde\|edcba |
| `Replicate` | Replicate padding: abcde -> aaaaa\|abcde\|eeeee |
| `Zero` | Zero padding: abcde -> 00000\|abcde\|00000 |

