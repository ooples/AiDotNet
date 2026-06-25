---
title: "ISpeakerDiarizer<T>"
description: "Interface for speaker diarization models that segment audio by speaker (\"who spoke when\")."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for speaker diarization models that segment audio by speaker ("who spoke when").

## For Beginners

Diarization is like labeling a transcript with "Speaker A said..."
"Speaker B said..." without knowing their names.

How it works:

1. Audio is segmented into small chunks
2. Speaker embeddings are extracted for each chunk
3. Clustering groups similar embeddings together
4. Each cluster represents a unique speaker
5. Output: Timeline showing when each speaker talks

Common use cases:

- Meeting transcription (separating participants)
- Podcast/interview processing
- Call center analytics
- Medical dictation

Challenges:

- Overlapping speech (multiple people talking at once)
- Short turns (quick back-and-forth conversation)
- Similar voices (e.g., siblings)
- Background noise and music

## How It Works

Speaker diarization partitions an audio stream into segments based on speaker identity.
It answers the question "Who spoke when?" without necessarily knowing who the speakers
are (unlike speaker identification which requires enrolled speakers).

This interface extends `IFullModel` for Tensor-based audio processing.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsOnnxMode` | Gets whether this model is running in ONNX inference mode. |
| `MinSegmentDuration` | Gets the minimum segment duration in seconds. |
| `SampleRate` | Gets the expected sample rate for input audio. |
| `SupportsOverlapDetection` | Gets whether this model can detect overlapping speech. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Diarize(Tensor<>,Nullable<Int32>,Int32,Int32)` | Performs speaker diarization on audio. |
| `DiarizeAsync(Tensor<>,Nullable<Int32>,Int32,Int32,CancellationToken)` | Performs speaker diarization asynchronously. |
| `DiarizeWithKnownSpeakers(Tensor<>,IReadOnlyList<SpeakerProfile<>>,Boolean)` | Performs diarization with known speaker profiles. |
| `ExtractSpeakerEmbeddings(Tensor<>,DiarizationResult<>)` | Gets speaker embeddings for each detected speaker. |
| `RefineDiarization(Tensor<>,DiarizationResult<>,)` | Refines diarization result by re-segmenting with different parameters. |

