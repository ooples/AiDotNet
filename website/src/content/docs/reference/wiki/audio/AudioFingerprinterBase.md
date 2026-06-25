---
title: "AudioFingerprinterBase<T>"
description: "Base class for audio fingerprinting algorithms."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Audio.Fingerprinting`

Base class for audio fingerprinting algorithms.

## For Beginners

Audio fingerprinting is like Shazam - it can identify a song
from a short audio clip, even if the audio is noisy or compressed.

How it works:

1. Audio is converted to a spectrogram
2. Key features (peaks, patterns) are extracted
3. Features are hashed into a compact fingerprint
4. Fingerprints can be matched against a database

This base class provides:

- Hash computation utilities
- Hamming distance for comparison
- Time alignment for matching

## How It Works

Audio fingerprinting creates compact identifiers that can recognize audio content
even after degradation (compression, noise, speed changes). Unlike neural network
approaches, fingerprinting typically uses signal processing techniques.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioFingerprinterBase` | Initializes a new instance of the AudioFingerprinterBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the hardware-accelerated computation engine for vectorized operations. |
| `FingerprintLength` | Gets the fingerprint length in bits or elements. |
| `FrameDuration` | Gets or sets the duration of each fingerprint frame in seconds. |
| `Name` | Gets the name of the fingerprinting algorithm. |
| `SampleRate` | Gets the expected sample rate for input audio. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAlignedSimilarity(UInt32[],UInt32[],Int32)` | Computes similarity at a specific alignment offset. |
| `ComputeAverageHammingDistance(UInt32[],UInt32[])` | Computes average Hamming distance between fingerprint hash sequences. |
| `ComputeHammingDistance(UInt32,UInt32)` | Computes Hamming distance between two fingerprint hashes. |
| `ComputeHashes([0:,0:],)` | Converts a feature matrix to hash values. |
| `ComputeSimilarity(AudioFingerprint<>,AudioFingerprint<>)` | Computes the similarity between two fingerprints. |
| `FindBestAlignment(UInt32[],UInt32[],Int32)` | Finds the best alignment offset between two fingerprint hash sequences. |
| `FindMatches(AudioFingerprint<>,AudioFingerprint<>,Int32)` | Finds matching segments between two fingerprints. |
| `Fingerprint(Tensor<>)` | Generates a fingerprint from audio data. |
| `Fingerprint(Vector<>)` | Generates a fingerprint from audio data. |
| `FrameToTime(Int32)` | Converts frame index to time in seconds. |
| `TimeToFrame(Double)` | Converts time in seconds to frame index. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Operations for the numeric type T. |

