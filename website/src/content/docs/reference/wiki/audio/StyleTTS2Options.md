---
title: "StyleTTS2Options"
description: "Configuration options for the StyleTTS 2 text-to-speech model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.TextToSpeech`

Configuration options for the StyleTTS 2 text-to-speech model.

## For Beginners

StyleTTS 2 is one of the most natural-sounding TTS models.
It works by separating what is said (content) from how it is said (style). You can
change the speaking style by providing a reference audio clip, or let the model
generate a natural style automatically. It uses a diffusion model (similar to image
generators like DALL-E) to create realistic-sounding prosody.

## How It Works

StyleTTS 2 (Li et al., 2023) uses diffusion models for style transfer and achieves
human-level naturalness on single-speaker synthesis (MOS 4.16 on LJSpeech). It
disentangles speech into content and style, allowing fine-grained control over
prosody, speaking rate, and emotion through style vectors.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `HopLength` | Gets or sets the hop length for spectrogram computation. |
| `IsMultiSpeaker` | Gets or sets whether the model is multi-speaker. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads in the text encoder. |
| `NumDiffusionSteps` | Gets or sets the number of diffusion steps for style generation. |
| `NumMels` | Gets or sets the number of mel-spectrogram frequency bins. |
| `NumTextEncoderLayers` | Gets or sets the number of text encoder layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ProsodyDim` | Gets or sets the prosody predictor hidden dimension. |
| `SampleRate` | Gets or sets the output audio sample rate in Hz. |
| `SpeakerEmbeddingDim` | Gets or sets the speaker embedding dimension for multi-speaker models. |
| `StyleDim` | Gets or sets the style encoder dimension. |
| `TextEncoderDim` | Gets or sets the text encoder hidden dimension. |
| `Variant` | Gets or sets the model variant ("base" or "large"). |

