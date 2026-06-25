---
title: "PANNsModel<T>"
description: "PANNs (Pretrained Audio Neural Networks) audio classifier — a CNN14-style convolutional backbone over log-mel spectrograms, trained for AudioSet tagging (Kong et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Fingerprinting`

PANNs (Pretrained Audio Neural Networks) audio classifier — a CNN14-style
convolutional backbone over log-mel spectrograms, trained for AudioSet
tagging (Kong et al. 2020).

## How It Works

PANNs CNN14 is the de-facto baseline transfer-learning model for audio
classification: four conv stages (64 → 128 → 256 → 512 channels), each
stage = Conv(3×3) + BN + ReLU + Conv(3×3) + BN + ReLU + AvgPool(2×2),
followed by a global average pool, a 2048-d embedding head, and a 527-
class linear classifier with sigmoid activation for multi-label output.
Trained on AudioSet (2 M weakly-labelled clips) at 32 kHz / 64-mel.

**Reference:** Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., &
Plumbley, M. D. (2020), "PANNs: Large-Scale Pretrained Audio Neural
Networks for Audio Pattern Recognition", IEEE/ACM TASLP.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PANNsModel(NeuralNetworkArchitecture<>,PANNsModelOptions)` | Initializes PANNs in native training / inference mode. |
| `PANNsModel(NeuralNetworkArchitecture<>,String,PANNsModelOptions)` | Initializes PANNs in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FingerprintLength` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildHannWindow(Int32)` | Builds a periodic Hann window of length `windowSize` as a `Tensor`: `w[n] = 0.5·(1 − cos(2πn/(N−1)))`. |
| `Classify(Tensor<>,Double)` | Classifies audio into AudioSet categories. |
| `ComputeSimilarity(AudioFingerprint<>,AudioFingerprint<>)` |  |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `FindMatches(AudioFingerprint<>,AudioFingerprint<>,Int32)` |  |
| `Fingerprint(Tensor<>)` |  |
| `Fingerprint(Vector<>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetTopK(Tensor<>,Int32)` | Returns the top-K class predictions sorted descending. |
| `InitializeLayers` | Initializes the CNN14 layer stack following the codebase's golden single-stream pattern: prefer user-supplied `Architecture.Layers`; otherwise fall back to `Double)`. |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessAudio(Tensor<>)` | Converts raw audio samples into a log-mel spectrogram via the engine's fused `Boolean)` kernel — the PANNs §3 64- mel × 320-hop pipeline routed through a single BLAS / GPU-eligible op. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateOptions(PANNsModelOptions)` | Validates that every option used in STFT / mel / CNN math is in range. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_hannWindow` | Cached Hann window for the STFT preprocessing step. |
| `_modelPath` | Path to the loaded ONNX model file when constructed in ONNX mode; `null` in native mode. |

