---
title: "IAudioVisualCorrespondenceModel<T>"
description: "Defines the contract for audio-visual correspondence learning models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for audio-visual correspondence learning models.

## For Beginners

Teaching AI to connect sounds with visuals!

Key capabilities:

- Sound source localization: Where in the image is the sound coming from?
- Audio-visual synchronization: Are the audio and video in sync?
- Cross-modal retrieval: Find images matching sounds and vice versa
- Audio-visual scene understanding: What's happening based on both modalities?

Examples:

- A dog barking → The model highlights the dog in the image
- Piano music → The model finds images of pianos
- Clapping sound → The model locates hands in the video

## How It Works

Audio-visual correspondence learning focuses on understanding the relationship
between what we see and what we hear. This enables tasks like finding the
source of a sound in a video, synchronizing audio and video, and understanding
audio-visual events.

## Properties

| Property | Summary |
|:-----|:--------|
| `AudioSampleRate` | Gets the expected audio sample rate. |
| `EmbeddingDimension` | Gets the embedding dimension for audio-visual features. |
| `VideoFrameRate` | Gets the expected video frame rate. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CheckSynchronization(Tensor<>,IEnumerable<Tensor<>>)` | Checks audio-visual synchronization. |
| `ClassifyScene(Tensor<>,IEnumerable<Tensor<>>,IEnumerable<String>)` | Classifies audio-visual scenes. |
| `ComputeCorrespondence(Tensor<>,IEnumerable<Tensor<>>)` | Computes audio-visual correspondence score. |
| `DescribeExpectedAudio(IEnumerable<Tensor<>>)` | Generates audio description from visual content. |
| `GetAudioEmbedding(Tensor<>,Int32)` | Computes audio embedding from waveform. |
| `GetVisualEmbedding(IEnumerable<Tensor<>>)` | Computes visual embedding from video frames. |
| `LearnCorrespondence(IEnumerable<Tensor<>>,IEnumerable<IEnumerable<Tensor<>>>,Int32)` | Learns correspondence from paired audio-visual data. |
| `LocalizeSoundSource(Tensor<>,IEnumerable<Tensor<>>)` | Localizes sound sources in video frames. |
| `RetrieveAudioFromVisuals(IEnumerable<Tensor<>>,IEnumerable<Vector<>>,Int32)` | Retrieves audio content matching visual input. |
| `RetrieveVisualsFromAudio(Tensor<>,IEnumerable<Vector<>>,Int32)` | Retrieves visual content matching audio. |
| `SeparateAudioByVisual(Tensor<>,Tensor<>)` | Separates audio sources based on visual guidance. |

