---
title: "RoomImpulseResponse<T>"
description: "Neural Room Impulse Response estimation model for acoustic analysis and dereverberation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Effects`

Neural Room Impulse Response estimation model for acoustic analysis and dereverberation.

## For Beginners

When you clap in a big room, you hear echoes. This model learns to
understand those echoes. Given a recording, it figures out the room's acoustic "fingerprint"
and can use it to remove room effects (dereverberation) or apply one room's sound to
another recording.

**Usage:**

## How It Works

Neural Room Impulse Response (RIR) estimation (2023-2024) uses deep learning to predict
acoustic characteristics of a room from audio recordings. The model estimates the RIR
encoding how sound propagates, reflects, and decays, enabling dereverberation, room
simulation, and acoustic environment matching.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RoomImpulseResponse(NeuralNetworkArchitecture<>,RoomImpulseResponseOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an RIR estimation model in native training mode. |
| `RoomImpulseResponse(NeuralNetworkArchitecture<>,String,RoomImpulseResponseOptions)` | Creates an RIR estimation model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EnhancementStrength` |  |
| `LatencySamples` |  |
| `NumChannels` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Enhance(Tensor<>)` |  |
| `EnhanceWithReference(Tensor<>,Tensor<>)` |  |
| `EstimateNoiseProfile(Tensor<>)` |  |
| `ProcessChunk(Tensor<>)` |  |
| `ResetState` |  |

