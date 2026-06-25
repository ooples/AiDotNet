---
title: "MatchaTTSOptions"
description: "Configuration options for the Matcha-TTS model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.TextToSpeech`

Configuration options for the Matcha-TTS model.

## For Beginners

Matcha-TTS is like a fast artist who can sketch a beautiful
portrait in just a few strokes. Other models (like Grad-TTS) need many small steps to
gradually refine the output, but Matcha-TTS takes a more direct path from noise to
speech - like drawing a straight line instead of a winding road.

Key advantages:

- Very fast: Only 2-4 synthesis steps (vs 50-1000 for diffusion models)
- Lightweight: Smaller model with fewer parameters
- High quality: Comparable to slower diffusion models
- Memory efficient: Uses optimal transport for efficient training

## How It Works

Matcha-TTS (Mehta et al., 2024) is a fast, lightweight TTS model based on conditional
flow matching (OT-CFM). Unlike diffusion-based TTS that requires many denoising steps,
Matcha-TTS generates high-quality mel-spectrograms in just 2-4 steps, achieving 10x
faster synthesis than Grad-TTS while maintaining comparable quality (MOS 4.04 on LJSpeech).

## Properties

| Property | Summary |
|:-----|:--------|
| `DecoderDim` | Gets or sets the decoder hidden dimension. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `DurationPredictorDim` | Gets or sets the duration predictor hidden dimension. |
| `HopLength` | Gets or sets the hop length for spectrogram alignment. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumDecoderLayers` | Gets or sets the number of decoder layers (U-Net blocks). |
| `NumDurationPredictorLayers` | Gets or sets the number of duration predictor layers. |
| `NumMels` | Gets or sets the number of mel-spectrogram frequency bins. |
| `NumSynthesisSteps` | Gets or sets the number of ODE solver steps for synthesis. |
| `NumTextEncoderHeads` | Gets or sets the number of attention heads in the text encoder. |
| `NumTextEncoderLayers` | Gets or sets the number of text encoder layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `PhonemeVocabSize` | Gets or sets the phoneme vocabulary size. |
| `SampleRate` | Gets or sets the output audio sample rate in Hz. |
| `Temperature` | Gets or sets the temperature for flow matching sampling. |
| `TextEncoderDim` | Gets or sets the text encoder hidden dimension. |
| `Variant` | Gets or sets the model variant ("small", "base"). |

