---
title: "ASTModel<T>"
description: "AST (Audio Spectrogram Transformer) — a single-stream Vision-Transformer applied to log-mel spectrograms, trained for audio event classification and fingerprinting (Gong et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Fingerprinting`

AST (Audio Spectrogram Transformer) — a single-stream Vision-Transformer
applied to log-mel spectrograms, trained for audio event classification
and fingerprinting (Gong et al. 2021).

## How It Works

AST applies the standard ViT recipe (Dosovitskiy et al. 2021) directly to
audio: convert the waveform to a log-mel spectrogram, treat it as a 2-D
image, slice it into 16×16 patches, embed each patch linearly, add
positional encodings, run N=12 transformer encoder layers, mean-pool, and
classify. AST-Base (768-dim, 12-layer, 12-head) initialised from a
DeiT/ViT ImageNet checkpoint achieves SOTA on AudioSet.

**Reference:** Gong, Y., Chung, Y.-A. & Glass, J. (2021),
"AST: Audio Spectrogram Transformer", Interspeech.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ASTModel(NeuralNetworkArchitecture<>,ASTModelOptions)` | Initializes AST in native training / inference mode. |
| `ASTModel(NeuralNetworkArchitecture<>,String,ASTModelOptions)` | Initializes AST in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FingerprintLength` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildHannWindow(Int32)` | Builds a periodic Hann window of length `windowSize` as a `Tensor`: `w[n] = 0.5·(1 − cos(2πn/(N−1)))`. |
| `Classify(Tensor<>,Int32)` | Returns the top-K class predictions with softmax-normalised probabilities. |
| `ComputeSimilarity(AudioFingerprint<>,AudioFingerprint<>)` |  |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `FindMatches(AudioFingerprint<>,AudioFingerprint<>,Int32)` |  |
| `Fingerprint(Tensor<>)` |  |
| `Fingerprint(Vector<>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the AST layer stack following the codebase's golden single-stream pattern: prefer user-supplied `Architecture.Layers`; otherwise fall back to the paper-faithful `Double)`. |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessAudio(Tensor<>)` | Converts raw audio samples into a log-mel spectrogram via the engine's fused `Boolean)` kernel — the AST §2.1 128-mel × 10 ms-hop pipeline routed through a single BLAS / GPU-eligible engine op. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SoftmaxLastAxis(Tensor<>)` | Numerically-stable softmax along the last axis. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateOptions(ASTModelOptions)` | Validates that every option used in STFT / mel / patch / transformer math is in range. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_hannWindow` | Cached Hann window for the STFT preprocessing step. |
| `_modelPath` | Path to the loaded ONNX model file when constructed in ONNX mode; `null` in native mode. |

