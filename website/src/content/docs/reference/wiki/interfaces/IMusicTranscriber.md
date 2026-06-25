---
title: "IMusicTranscriber<T>"
description: "Defines the contract for automatic music transcription (audio to notes)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for automatic music transcription (audio to notes).

## For Beginners

Music transcription is like having a computer "listen" to music
and write down the notes. The output is similar to sheet music data:

- Which note is playing (e.g., C4, A#3)
- When each note starts and stops
- How loud each note is (velocity)

This is used for:

- Converting recordings to MIDI files
- Music education (showing what notes are played)
- Music analysis and research
- Karaoke systems

## How It Works

Music transcription converts audio recordings into symbolic note representations (like MIDI).
It detects what notes are played, when they start and end, and optionally their velocity.

## Properties

| Property | Summary |
|:-----|:--------|
| `MidiOffset` | Gets the MIDI offset (lowest MIDI note number, e.g., 21 for A0). |
| `NumMidiNotes` | Gets the number of MIDI notes this transcriber can detect. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractNotes(Tensor<>,Tensor<>,Double,Double)` | Extracts notes from frame and onset activations using post-processing. |
| `GetFrameActivations(Tensor<>)` | Gets frame-level note activations (piano roll representation). |
| `GetOnsetActivations(Tensor<>)` | Gets frame-level onset activations. |
| `Transcribe(Tensor<>)` | Transcribes audio into a list of detected notes. |
| `TranscribeAsync(Tensor<>,CancellationToken)` | Transcribes audio asynchronously. |

