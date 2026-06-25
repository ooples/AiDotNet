---
title: "MT3Options"
description: "Configuration options for the MT3 (Multi-Track Music Transcription) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.MusicAnalysis`

Configuration options for the MT3 (Multi-Track Music Transcription) model.

## For Beginners

MT3 listens to a full song with multiple instruments and writes out
the sheet music (as MIDI) for each instrument separately. It can tell which notes the piano
is playing while also transcribing the guitar, drums, and bass at the same time.

## How It Works

MT3 (Gardner et al., 2022, Google) is a Transformer-based model that transcribes polyphonic
audio into MIDI across multiple instruments simultaneously. It uses a T5-style encoder-decoder
architecture with spectrogram input and tokenized MIDI output.

## Properties

| Property | Summary |
|:-----|:--------|
| `DecoderDim` | Gets or sets the decoder hidden dimension. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderDim` | Gets or sets the encoder hidden dimension. |
| `FftSize` | Gets or sets the FFT window size. |
| `HopLength` | Gets or sets the hop length between frames. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxInstruments` | Gets or sets the maximum number of instruments. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumDecoderLayers` | Gets or sets the number of decoder layers. |
| `NumEncoderLayers` | Gets or sets the number of encoder layers. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `VocabSize` | Gets or sets the MIDI vocabulary size (tokens). |

