---
title: "CLAPModel<T>"
description: "CLAP (Contrastive Language-Audio Pretraining) — a dual-encoder neural network that learns to align audio and text representations in a shared embedding space via a contrastive objective."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Fingerprinting`

CLAP (Contrastive Language-Audio Pretraining) — a dual-encoder neural network
that learns to align audio and text representations in a shared embedding
space via a contrastive objective.

## How It Works

CLAP (Wu et al. 2023) trains an audio encoder and a text encoder so that
matching (audio, caption) pairs produce nearby embeddings and unrelated
pairs produce distant embeddings. The audio side is HTSAT (Chen et al. 2022),
a hierarchical Swin Transformer (Liu et al. 2021) over mel-spectrogram
patches. The text side is a RoBERTa-style transformer stack (Liu et al. 2019)
over BPE token IDs. A learnable temperature τ scales the cosine-similarity
logits during contrastive training (CLIP / CLAP convention).

**Capabilities**:

- Zero-shot audio classification with text prompts
- Audio-to-text and text-to-audio retrieval
- Semantic audio fingerprinting via the projection head

**Reference:** Wu, Y. et al. (2023), "Large-Scale Contrastive Language-Audio
Pre-Training with Feature Fusion and Keyword-to-Caption Augmentation", ICASSP.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CLAPModel(NeuralNetworkArchitecture<>,CLAPModelOptions)` | Initializes a new instance of `CLAPModel` in native training / inference mode. |
| `CLAPModel(NeuralNetworkArchitecture<>,String,String,CLAPModelOptions)` | Initializes a new instance of `CLAPModel` in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FingerprintLength` |  |
| `Name` |  |
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildHannWindow(Int32)` | Builds a periodic Hann window of length `windowSize` as a `Tensor`: `w[n] = 0.5·(1 − cos(2πn/(N−1)))`. |
| `ComputeCosineSimilarity(Tensor<>,Tensor<>)` | Cosine similarity between two row-aligned embeddings (already L2-normalised). |
| `ComputeSimilarity(AudioFingerprint<>,AudioFingerprint<>)` |  |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `EncodeAudio(Tensor<>)` | Encodes audio into a CLAP embedding vector in the shared text-audio space. |
| `EncodeText(Int32[])` | Convenience overload: tokenise + encode. |
| `EncodeText(Tensor<>)` | Encodes a tokenised text caption into a CLAP embedding vector. |
| `FindMatches(AudioFingerprint<>,AudioFingerprint<>,Int32)` |  |
| `Fingerprint(Tensor<>)` |  |
| `Fingerprint(Vector<>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `InitializeLayers` | Initializes the neural network layers following the codebase's golden dual-stream pattern. |
| `L2Normalize(Tensor<>)` | L2-normalises along the last (feature) axis so cosine similarity reduces to a plain dot product downstream. |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessAudio(Tensor<>)` | Converts raw audio samples into a log-mel spectrogram via the engine's fused `Boolean)` kernel (Hann window → STFT → triangular HTK mel filterbank → log power). |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SymmetricRowCrossEntropy(Tensor<>,Int32)` | Computes 0.5 * mean cross-entropy of softmax(logits) against the diagonal target (positive pair at the same row index), tape-tracked via engine ops. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ZeroShotClassify(Tensor<>,String[],Func<String,Int32[]>)` | Performs zero-shot audio classification: rank each text prompt by its CLAP-similarity to the supplied audio clip. |

