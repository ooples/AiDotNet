---
title: "SoundLocalizerOptions"
description: "Options for sound source localization."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Localization`

Options for sound source localization.

## For Beginners

These options configure the SoundLocalizer model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SoundLocalizerOptions` | Initializes a new instance with default values. |
| `SoundLocalizerOptions(SoundLocalizerOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Algorithm` | Localization algorithm. |
| `AngleResolution` | Angular resolution in degrees. |
| `CenterFrequency` | Center frequency for narrowband processing. |
| `FrameSize` | Frame size for MUSIC algorithm. |
| `ModelPath` | Path to ONNX model file (optional). |
| `OnnxOptions` | ONNX model options. |
| `SampleRate` | Audio sample rate. |
| `SpeedOfSound` | Speed of sound in m/s. |

