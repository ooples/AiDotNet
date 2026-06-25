---
title: "RoomImpulseResponseOptions"
description: "Configuration options for the Room Impulse Response (RIR) estimation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Effects`

Configuration options for the Room Impulse Response (RIR) estimation model.

## For Beginners

When you clap in a big room, you hear echoes. This model learns to
understand those echoes. Given a recording, it can figure out the room's acoustic "fingerprint"
and use it to remove room effects (dereverberation) or apply one room's sound to another recording.

## How It Works

Neural Room Impulse Response estimation (2023-2024) uses deep learning to predict the
acoustic characteristics of a room from audio recordings. The model estimates the RIR
which encodes how sound propagates, reflects, and decays in a given space, enabling
applications like dereverberation, room simulation, and acoustic environment matching.

## Properties

| Property | Summary |
|:-----|:--------|
| `DereverberationStrength` | Gets or sets the dereverberation strength (0-1). |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderDim` | Gets or sets the encoder dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumEncoderLayers` | Gets or sets the number of encoder layers. |
| `NumFrequencyBins` | Gets or sets the number of frequency bins for spectral estimation. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `RIRLength` | Gets or sets the RIR length in samples. |
| `RT60WindowSeconds` | Gets or sets the RT60 estimation window in seconds. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Variant` | Gets or sets the model variant. |

