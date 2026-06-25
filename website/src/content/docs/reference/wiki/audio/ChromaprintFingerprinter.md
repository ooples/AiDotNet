---
title: "ChromaprintFingerprinter<T>"
description: "Chromaprint-style audio fingerprinter based on chroma features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Fingerprinting`

Chromaprint-style audio fingerprinter based on chroma features.

## For Beginners

Chromaprint works by analyzing the musical notes
present in the audio. It groups all octaves of the same note together (C1, C2, C3
all become "C") and tracks how these change over time. This makes it good at
matching different recordings of the same song.

## How It Works

This fingerprinter uses chromagram analysis similar to the Chromaprint algorithm
used by AcoustID. It extracts chroma features and converts them to a compact
binary representation that is robust to tempo changes and transposition.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChromaprintFingerprinter(ChromaprintOptions)` | Creates a new Chromaprint fingerprinter. |

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

