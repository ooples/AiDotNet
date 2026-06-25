---
title: "NsynthDataLoaderOptions"
description: "Configuration for the NSynth (Neural Synth) audio dataset loader (Engel et al."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Audio.Benchmarks`

Configuration for the NSynth (Neural Synth) audio dataset loader (Engel et al. 2017).

## How It Works

NSynth — 305,979 musical notes from 1,006 instruments, each as a 4-second
16 kHz mono WAV. Annotated with pitch, velocity, instrument family (11 classes),
instrument source (3 classes), and qualities. Standard benchmark for
audio synthesis / pitch-conditional generation.

## Properties

| Property | Summary |
|:-----|:--------|
| `SampleRate` | Source-file sample rate. |
| `Samples` | Samples per clip (4 sec * 16 kHz = 64,000). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

