---
title: "OnsetsAndFramesOptions"
description: "Configuration options for the Onsets and Frames piano transcription model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.MusicAnalysis`

Configuration options for the Onsets and Frames piano transcription model.

## For Beginners

Onsets and Frames listens to piano music and writes down which notes
are being played and when. It detects two things: "onsets" (when a key is pressed) and
"frames" (which keys are held down at each moment). By combining both, it produces accurate
note-by-note transcriptions of piano recordings.

## How It Works

Onsets and Frames (Hawthorne et al., 2018, Google Magenta) jointly predicts onsets and
frame-level note activations for automatic piano transcription. The model uses a CNN front-end
with bidirectional LSTMs, and was trained on the MAESTRO dataset. It achieves frame-level
note F1 of ~90% on piano recordings.

## Properties

| Property | Summary |
|:-----|:--------|
| `AcousticModelDim` | Gets or sets the acoustic model CNN feature dimension. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `FMax` | Gets or sets the maximum frequency for the mel filterbank. |
| `FMin` | Gets or sets the minimum frequency for the mel filterbank. |
| `FftSize` | Gets or sets the FFT window size in samples. |
| `FrameThreshold` | Gets or sets the frame (note held) detection threshold. |
| `HopLength` | Gets or sets the hop length between frames in samples. |
| `LearningRate` | Gets or sets the learning rate for training. |
| `LstmHiddenSize` | Gets or sets the bidirectional LSTM hidden size. |
| `MidiOffset` | Gets or sets the lowest MIDI note (A0 = 21). |
| `MinNoteDurationSec` | Gets or sets the minimum note duration in seconds. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumLstmLayers` | Gets or sets the number of LSTM layers. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `NumMidiNotes` | Gets or sets the number of MIDI notes (88 piano keys: A0=21 to C8=108). |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `OnsetThreshold` | Gets or sets the onset detection threshold. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |

