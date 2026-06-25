---
title: "IAudioFingerprinter<T>"
description: "Interface for audio fingerprinting algorithms."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for audio fingerprinting algorithms.

## For Beginners

An audio fingerprint is like a "signature" for
a piece of audio. Just like human fingerprints identify individuals, audio
fingerprints identify songs or sound recordings. Services like Shazam use
fingerprinting to identify songs from short recordings.

## How It Works

Audio fingerprinting creates compact representations of audio that can be
used for identification and similarity matching. Different algorithms
trade off between accuracy, speed, and robustness to transformations.

## Properties

| Property | Summary |
|:-----|:--------|
| `FingerprintLength` | Gets the fingerprint length in bits or elements. |
| `Name` | Gets the name of the fingerprinting algorithm. |
| `SampleRate` | Gets the expected sample rate for input audio. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSimilarity(AudioFingerprint<>,AudioFingerprint<>)` | Computes the similarity between two fingerprints. |
| `FindMatches(AudioFingerprint<>,AudioFingerprint<>,Int32)` | Finds matching segments between two fingerprints. |
| `Fingerprint(Tensor<>)` | Generates a fingerprint from audio data. |
| `Fingerprint(Vector<>)` | Generates a fingerprint from audio data. |

