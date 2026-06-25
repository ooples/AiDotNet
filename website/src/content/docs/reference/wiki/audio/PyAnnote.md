---
title: "PyAnnote<T>"
description: "pyannote 3.x end-to-end speaker diarization model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Speaker`

pyannote 3.x end-to-end speaker diarization model.

## For Beginners

pyannote figures out "who spoke when" in a recording with
multiple speakers. It's like automatically labeling a meeting transcript with
"Speaker A: 0:00-0:15, Speaker B: 0:15-0:45..." It can even detect when two people
talk at the same time (overlapping speech).

**Usage:**

## How It Works

pyannote.audio 3.x (Plaquet & Bredin, ASRU 2023) is a state-of-the-art speaker
diarization pipeline using end-to-end neural segmentation with PyanNet architecture.
It segments audio into speaker turns, supports overlapping speech detection, and
achieves 11.2% DER on AMI Mix-Headset benchmark.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PyAnnote(NeuralNetworkArchitecture<>,PyAnnoteOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a pyannote diarization model in native training mode. |
| `PyAnnote(NeuralNetworkArchitecture<>,String,PyAnnoteOptions)` | Creates a pyannote diarization model in ONNX inference mode. |

