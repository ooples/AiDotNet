---
title: "PyAnnoteOptions"
description: "Configuration options for the pyannote 3.x speaker diarization model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Speaker`

Configuration options for the pyannote 3.x speaker diarization model.

## For Beginners

pyannote is a system that figures out "who spoke when" in a
recording with multiple speakers. It splits audio into segments, assigns each to a
speaker, and can even detect when two people talk at the same time. It's widely used
for meeting transcription, podcast processing, and call analytics.

## How It Works

pyannote.audio 3.x (Plaquet & Bredin, ASRU 2023) is a state-of-the-art speaker
diarization pipeline using end-to-end neural segmentation with PyanNet architecture.
It segments audio into speaker turns and supports overlapping speech detection.
Achieves 11.2% DER on AMI Mix-Headset benchmark.

## Properties

| Property | Summary |
|:-----|:--------|
| `ChunkDurationSeconds` | Gets or sets the segmentation chunk duration in seconds. |
| `ChunkStepSeconds` | Gets or sets the step between consecutive chunks in seconds. |
| `ClusteringThreshold` | Gets or sets the clustering threshold for speaker assignment. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDim` | Gets or sets the embedding dimension for speaker embeddings. |
| `EmbeddingModelPath` | Gets or sets the path to a separate embedding model. |
| `EnableOverlapDetection` | Gets or sets whether overlapping speech detection is enabled. |
| `FftSize` | Gets or sets the FFT window size in samples. |
| `HopLength` | Gets or sets the hop length between frames in samples. |
| `LSTMHiddenSize` | Gets or sets the LSTM hidden size for the segmentation model. |
| `LearningRate` | Gets or sets the learning rate for training. |
| `LinearDim` | Gets or sets the linear layer hidden dimension after LSTM. |
| `MaxSpeakers` | Gets or sets the maximum number of speakers (null for auto-detection). |
| `MaxSpeakersPerChunk` | Gets or sets the maximum number of speakers per chunk. |
| `MinSegmentDuration` | Gets or sets the minimum segment duration in seconds. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumLSTMLayers` | Gets or sets the number of LSTM layers. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `SincNetFilters` | Gets or sets the number of SincNet filters in the first layer. |
| `WeightDecay` | Gets or sets the weight decay for regularization. |

