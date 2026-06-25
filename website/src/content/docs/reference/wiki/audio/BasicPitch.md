---
title: "BasicPitch<T>"
description: "Basic Pitch polyphonic music transcription model from Spotify."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.MusicAnalysis`

Basic Pitch polyphonic music transcription model from Spotify.

## For Beginners

Basic Pitch is like a music-to-MIDI converter. You feed it a recording
of music (even with multiple instruments playing at once) and it outputs a list of every note
that was played, when it started, when it stopped, and how loud it was. This is called
"polyphonic transcription" because it handles multiple notes at the same time.

**Usage:**

## How It Works

Basic Pitch (Bittner et al., 2022) is a lightweight CNN for polyphonic music transcription.
It produces three outputs: note activations, onset activations, and pitch contour. Combined,
these produce MIDI-like note events. The model is fast enough for real-time use and handles
multiple simultaneous instruments.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BasicPitch(NeuralNetworkArchitecture<>,BasicPitchOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Basic Pitch model in native training mode. |
| `BasicPitch(NeuralNetworkArchitecture<>,String,BasicPitchOptions)` | Creates a Basic Pitch model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MidiOffset` |  |
| `NumMidiNotes` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractNotes(Tensor<>,Tensor<>,Double,Double)` |  |
| `GetFrameActivations(Tensor<>)` |  |
| `GetOnsetActivations(Tensor<>)` |  |
| `Transcribe(Tensor<>)` |  |
| `TranscribeAsync(Tensor<>,CancellationToken)` |  |

