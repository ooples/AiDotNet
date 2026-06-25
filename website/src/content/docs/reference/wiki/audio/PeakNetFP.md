---
title: "PeakNetFP<T>"
description: "PeakNetFP spectral peak-based neural audio fingerprinting model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Fingerprinting`

PeakNetFP spectral peak-based neural audio fingerprinting model.

## For Beginners

PeakNetFP identifies songs by finding the "peaks" in their sound
spectrum (like the loudest frequencies at each moment) and then uses AI to turn those peaks
into a compact code. It's a hybrid approach combining classical peak picking with neural encoding.

**Usage:**

## How It Works

PeakNetFP combines traditional spectral peak picking with a neural network for robust audio
fingerprinting. It detects spectral peaks in the spectrogram and uses a CNN to encode peak
constellations into compact binary hashes, offering both speed and robustness.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PeakNetFP(NeuralNetworkArchitecture<>,PeakNetFPOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a PeakNetFP model in native training mode. |
| `PeakNetFP(NeuralNetworkArchitecture<>,String,PeakNetFPOptions)` | Creates a PeakNetFP model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FingerprintLength` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSimilarity(AudioFingerprint<>,AudioFingerprint<>)` |  |
| `FindMatches(AudioFingerprint<>,AudioFingerprint<>,Int32)` |  |
| `Fingerprint(Tensor<>)` |  |
| `Fingerprint(Vector<>)` |  |

