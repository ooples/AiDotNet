---
title: "OffsetNoiseSchedule<T>"
description: "Offset noise schedule that adds a global offset to noise for improved dark/bright image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers.NoiseSchedules`

Offset noise schedule that adds a global offset to noise for improved dark/bright image generation.

## For Beginners

Standard diffusion models struggle to generate very dark or very
bright images because the noise always averages out to medium brightness. Offset noise
fixes this by occasionally adding brightness shifts to the entire image during training.

## How It Works

Standard diffusion noise is zero-mean per pixel, which biases the model toward
mid-tone images. Offset noise adds a small per-channel offset (same value for all
pixels in a channel), enabling the model to generate very dark or very bright images.

Reference: Originally proposed by Nicholas Guttenberg, widely adopted in SD community

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OffsetNoiseSchedule(Double)` | Initializes a new instance with the specified offset strength. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyOffset(Vector<>,Int32,Random)` | Applies offset noise to a standard Gaussian noise vector. |

