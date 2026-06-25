---
title: "Zipformer<T>"
description: "Zipformer speech recognition model (Yao et al., 2023, Next-gen Kaldi)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.SpeechRecognition`

Zipformer speech recognition model (Yao et al., 2023, Next-gen Kaldi).

## For Beginners

Zipformer is an improved version of the Conformer that's both faster
and more accurate. It processes speech at different "zoom levels":

- Some layers look at fine details (individual sounds at full resolution)
- Middle layers zoom out (every 2nd, 4th, or 8th frame) for bigger-picture patterns
- Then it zooms back in to combine everything

This U-Net-like structure (like looking through a microscope at different magnifications)
makes it one of the most efficient speech encoders available today.

**Usage:**

## How It Works

Zipformer (Yao et al., 2023) is a more efficient variant of the Conformer with a U-Net-like
structure that processes speech at different temporal resolutions. It uses BiasNorm instead
of LayerNorm, SwooshR/SwooshL activations, and temporal downsampling across encoder stacks.
This achieves better accuracy with fewer parameters than standard Conformer, with WER 2.0%/4.4%
on LibriSpeech test-clean/other without an external language model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Zipformer(NeuralNetworkArchitecture<>,String,ZipformerOptions)` | Creates a Zipformer model in ONNX inference mode. |
| `Zipformer(NeuralNetworkArchitecture<>,ZipformerOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Zipformer model in native training mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportedLanguages` |  |
| `SupportsStreaming` |  |
| `SupportsWordTimestamps` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectLanguage(Tensor<>)` |  |
| `DetectLanguageProbabilities(Tensor<>)` |  |
| `StartStreamingSession(String)` |  |
| `Transcribe(Tensor<>,String,Boolean)` |  |
| `TranscribeAsync(Tensor<>,String,Boolean,CancellationToken)` |  |

