---
title: "BasicPitchOptions"
description: "Configuration options for the Basic Pitch multi-pitch detection model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.MusicAnalysis`

Configuration options for the Basic Pitch multi-pitch detection model.

## For Beginners

Basic Pitch turns audio of music into "sheet music" data. It can
detect multiple notes playing at the same time (like a piano chord), when each note starts
and stops, and what pitch each note is. The output is similar to MIDI - a list of
(start_time, end_time, pitch, velocity) for every detected note.

## How It Works

Basic Pitch (Bittner et al., 2022) from Spotify is a lightweight neural network for
polyphonic music transcription. It detects note onsets, offsets, and pitch for multiple
simultaneous notes, producing MIDI-like output from audio. Unlike monophonic pitch detectors
(CREPE), Basic Pitch handles chords and polyphonic music.

## Properties

| Property | Summary |
|:-----|:--------|
| `BinsPerOctave` | Gets or sets the number of frequency bins per octave. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderFilters` | Gets or sets the number of convolutional filters in the encoder. |
| `HopLength` | Gets or sets the hop length for the CQT in samples. |
| `LearningRate` | Gets or sets the learning rate for training. |
| `MaxFrequencyHz` | Gets or sets the maximum frequency in Hz for note detection. |
| `MidiOffset` | Gets or sets the lowest MIDI note (A0 = 21). |
| `MinFrequencyHz` | Gets or sets the minimum frequency in Hz for note detection. |
| `MinNoteDurationSec` | Gets or sets the minimum note duration in seconds. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NoteThreshold` | Gets or sets the note (frame) detection threshold. |
| `NumEncoderLayers` | Gets or sets the number of convolutional layers in the encoder. |
| `NumHarmonicBins` | Gets or sets the number of harmonically-stacked CQT bins. |
| `NumHarmonics` | Gets or sets the number of harmonics to stack. |
| `NumMidiNotes` | Gets or sets the number of MIDI notes to predict (88 piano keys by default, A0 to C8). |
| `NumOutputHeads` | Gets or sets the number of output heads (note, onset, contour). |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `OnsetThreshold` | Gets or sets the onset detection threshold. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |

