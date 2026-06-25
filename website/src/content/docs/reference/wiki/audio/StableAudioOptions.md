---
title: "StableAudioOptions"
description: "Configuration options for Stable Audio generation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.StableAudio`

Configuration options for Stable Audio generation.

## For Beginners

Stable Audio generates professional-quality audio:

Example prompts:

- "Upbeat electronic dance track with synth leads and heavy bass drop"
- "Peaceful ambient soundscape with soft pads and nature sounds"
- "Epic orchestral trailer music with dramatic brass and percussion"
- "Lo-fi hip hop beat with jazzy piano chords and vinyl crackle"

Tips for good prompts:

- Be specific about genre, instruments, mood, and tempo
- Mention audio characteristics (stereo width, dynamics)
- Include style references when appropriate

## How It Works

Stable Audio is Stability AI's state-of-the-art audio generation model using
latent diffusion with a Diffusion Transformer (DiT) architecture. It supports
high-quality music and sound effects generation with variable-length output.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StableAudioOptions` | Initializes a new instance with default values. |
| `StableAudioOptions(StableAudioOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DitHiddenDim` | Gets or sets the DiT hidden dimension. |
| `DitPath` | Gets or sets the path to the DiT denoiser ONNX model. |
| `DropoutRate` | Gets or sets the dropout rate for training. |
| `DurationSeconds` | Gets or sets the default duration of generated audio in seconds. |
| `GuidanceScale` | Gets or sets the classifier-free guidance scale. |
| `LatentDimension` | Gets or sets the latent dimension. |
| `MaxAudioLength` | Gets or sets the maximum audio latent length. |
| `MaxDurationSeconds` | Gets or sets the maximum duration in seconds. |
| `MaxTextLength` | Gets or sets the maximum text sequence length. |
| `ModelSize` | Gets or sets the model size variant. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumDitBlocks` | Gets or sets the number of DiT blocks. |
| `NumInferenceSteps` | Gets or sets the number of diffusion steps. |
| `OnnxOptions` | Gets or sets the ONNX execution options. |
| `SampleRate` | Gets or sets the output sample rate in Hz. |
| `Stereo` | Gets or sets whether to generate stereo audio. |
| `TextEmbeddingDim` | Gets or sets the T5 embedding dimension. |
| `TextEncoderPath` | Gets or sets the path to the T5 text encoder ONNX model. |
| `TimingConditioningScale` | Gets or sets the conditioning scale for timing information. |
| `VaePath` | Gets or sets the path to the VAE ONNX model. |

