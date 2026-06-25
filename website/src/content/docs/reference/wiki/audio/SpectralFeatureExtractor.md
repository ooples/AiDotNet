---
title: "SpectralFeatureExtractor<T>"
description: "Extracts spectral features from audio signals including centroid, bandwidth, rolloff, and flux."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Features`

Extracts spectral features from audio signals including centroid, bandwidth, rolloff, and flux.

## For Beginners

These features describe "what the sound looks like" in terms of frequency:

- **Spectral Centroid**: The "center of mass" of the spectrum - high for bright sounds (cymbals),

low for dull sounds (bass drum). Think of it as the "brightness" of the sound.

- **Spectral Bandwidth**: How spread out the frequencies are. Wide for rich sounds (orchestra),

narrow for pure tones (flute).

- **Spectral Rolloff**: The frequency below which most (e.g., 85%) of the energy is concentrated.

Useful for distinguishing voiced from unvoiced speech.

- **Spectral Flux**: How much the spectrum changes between frames. High during transients (drum hits),

low during sustained sounds.

- **Spectral Flatness**: How "noisy" vs "tonal" the sound is. 1.0 = pure noise, 0.0 = pure tone.

Usage:

## How It Works

Spectral features describe the shape and characteristics of an audio signal's
frequency content. They are widely used for audio classification, music analysis,
and speech processing.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpectralFeatureExtractor(SpectralFeatureOptions)` | Initializes a new spectral feature extractor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureDimension` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Extract(Tensor<>)` |  |
| `GetFeatureIndex(SpectralFeatureType)` | Gets the column index for a specific feature type in the extracted feature tensor. |

