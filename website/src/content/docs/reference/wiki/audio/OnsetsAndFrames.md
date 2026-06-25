---
title: "OnsetsAndFrames<T>"
description: "Onsets and Frames piano transcription model from Google Magenta."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.MusicAnalysis`

Onsets and Frames piano transcription model from Google Magenta.

## For Beginners

Onsets and Frames is a piano-specific music transcriber. It listens
to piano music and detects every key press: when each key is pressed (onset), how long it
is held (frame), and which key it is (pitch). The output is a list of notes that can be
saved as MIDI or displayed as sheet music.

**Usage:**

## How It Works

Onsets and Frames (Hawthorne et al., 2018) jointly predicts note onsets and frame-level
activations for automatic piano transcription. The model uses CNN acoustic features with
bidirectional LSTMs, trained on the MAESTRO dataset. It achieves ~90% note F1 on piano.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OnsetsAndFrames(NeuralNetworkArchitecture<>,OnsetsAndFramesOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an Onsets and Frames model in native training mode. |
| `OnsetsAndFrames(NeuralNetworkArchitecture<>,String,OnsetsAndFramesOptions)` | Creates an Onsets and Frames model in ONNX inference mode. |

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

