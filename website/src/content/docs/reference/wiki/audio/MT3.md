---
title: "MT3<T>"
description: "MT3 (Multi-Track Music Transcription) model using T5-style encoder-decoder architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.MusicAnalysis`

MT3 (Multi-Track Music Transcription) model using T5-style encoder-decoder architecture.

## For Beginners

MT3 listens to a full song with multiple instruments and writes out
the sheet music (as MIDI) for each instrument separately. It can tell which notes the piano
is playing while also transcribing the guitar, drums, and bass at the same time.

**Usage:**

## How It Works

MT3 (Gardner et al., 2022, Google) is a Transformer-based model that transcribes polyphonic
audio into MIDI across multiple instruments simultaneously. It uses a T5-style encoder-decoder
architecture with spectrogram input and tokenized MIDI output, achieving state-of-the-art
multi-instrument transcription on the Slakh2100 dataset.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MT3(NeuralNetworkArchitecture<>,MT3Options,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an MT3 model in native training mode. |
| `MT3(NeuralNetworkArchitecture<>,String,MT3Options)` | Creates an MT3 model in ONNX inference mode. |

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

