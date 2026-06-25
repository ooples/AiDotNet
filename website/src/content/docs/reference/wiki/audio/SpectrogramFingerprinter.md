---
title: "SpectrogramFingerprinter<T>"
description: "Spectrogram peak-based audio fingerprinter (Shazam-style)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Fingerprinting`

Spectrogram peak-based audio fingerprinter (Shazam-style).

## For Beginners

This algorithm finds the loudest frequency points
in the audio (like mountain peaks on a landscape) and remembers their positions.
By comparing peak patterns, it can identify songs even with background noise
or slight speed changes.

## How It Works

This fingerprinter uses spectral peak detection similar to the Shazam algorithm.
It finds prominent peaks in the spectrogram and creates hash codes from peak
pairs, providing robustness to noise and speed variations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpectrogramFingerprinter(SpectrogramFingerprintOptions)` | Creates a new spectrogram-based fingerprinter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of the fingerprinting algorithm. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSimilarity(AudioFingerprint<>,AudioFingerprint<>)` | Computes similarity between two fingerprints. |
| `FindMatches(AudioFingerprint<>,AudioFingerprint<>,Int32)` | Finds matching segments between fingerprints. |
| `Fingerprint(Tensor<>)` | Generates a fingerprint from audio tensor. |
| `Fingerprint(Vector<>)` | Generates a fingerprint from audio vector. |

