---
title: "ZipformerOptions"
description: "Configuration options for the Zipformer speech recognition model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.SpeechRecognition`

Configuration options for the Zipformer speech recognition model.

## For Beginners

Zipformer is an improved version of the Conformer that's both faster
and more accurate. It processes speech at different "zoom levels" - some parts look at fine
details (individual sounds) and other parts look at the bigger picture (whole words and phrases).
This makes it one of the most efficient speech encoders available.

## How It Works

Zipformer (Yao et al., 2023, Next-gen Kaldi) is a more efficient variant of the Conformer
with temporal downsampling, BiasNorm instead of LayerNorm, and SwooshR/SwooshL activations.
It uses a U-Net-like structure with different time resolutions at different encoder stacks,
achieving better accuracy with fewer parameters than standard Conformer.

## Properties

| Property | Summary |
|:-----|:--------|
| `DownsampleFactors` | Gets or sets the downsampling factors per stack. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderDims` | Gets or sets the encoder dimensions at each stack (U-Net style). |
| `Language` | Gets or sets the language code. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumHeadsPerStack` | Gets or sets the attention heads per stack. |
| `NumLayersPerStack` | Gets or sets the number of layers per encoder stack. |
| `NumMels` | Gets or sets the number of mel spectrogram channels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Variant` | Gets or sets the model variant ("small", "medium", "large"). |
| `VocabSize` | Gets or sets the vocabulary size. |
| `Vocabulary` | Gets or sets the CTC vocabulary (characters or BPE tokens). |

