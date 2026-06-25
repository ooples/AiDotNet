---
title: "BranchformerOptions"
description: "Configuration options for the Branchformer speech recognition model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.SpeechRecognition.ConformerFamily`

Configuration options for the Branchformer speech recognition model.

## How It Works

The Branchformer (Peng et al., 2022) uses parallel branches: one for self-attention
and one for a convolutional gating MLP (cgMLP), merged via learned concatenation.
This design captures both global and local dependencies in parallel, achieving
competitive or better accuracy than Conformer with similar computational cost.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BranchformerOptions` | Initializes a new instance with default values. |
| `BranchformerOptions(BranchformerOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CgmlpDim` | Gets or sets the cgMLP intermediate dimension. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderDim` | Gets or sets the encoder hidden dimension. |
| `Language` | Gets or sets the default language code. |
| `MaxAudioLengthSeconds` | Gets or sets the maximum audio length in seconds. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumEncoderLayers` | Gets or sets the number of Branchformer encoder layers. |
| `NumMels` | Gets or sets the number of mel-spectrogram frequency bins. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `VocabSize` | Gets or sets the vocabulary size for the CTC output head. |
| `Vocabulary` | Gets or sets the CTC vocabulary. |

