---
title: "AudioLDMOptions"
description: "Configuration options for AudioLDM text-to-audio generation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.AudioLDM`

Configuration options for AudioLDM text-to-audio generation.

## For Beginners

AudioLDM generates realistic audio from descriptions:

Example prompts:

- "A dog barking followed by children laughing"
- "Rain falling on a tin roof with distant thunder"
- "Footsteps on gravel approaching and stopping"
- "Piano music in a concert hall with audience applause"

Tips for good prompts:

- Be specific about the sound source and environment
- Include temporal information (before, after, while)
- Mention acoustic properties (loud, soft, distant, echoing)

## How It Works

AudioLDM is a latent diffusion model for text-to-audio generation. It operates
in a compressed latent space learned by a VAE, making generation efficient while
maintaining high audio quality.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioLDMOptions` | Initializes a new instance with default values. |
| `AudioLDMOptions(AudioLDMOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClapEmbeddingDim` | Gets or sets the CLAP embedding dimension. |
| `ClapEncoderPath` | Gets or sets the path to the CLAP text encoder ONNX model. |
| `DropoutRate` | Gets or sets the dropout rate for training. |
| `DurationSeconds` | Gets or sets the default duration of generated audio in seconds. |
| `GuidanceScale` | Gets or sets the classifier-free guidance scale. |
| `HopLength` | Gets or sets the hop length for spectrogram computation. |
| `LatentDimension` | Gets or sets the VAE latent dimension. |
| `LatentDownsampleFactor` | Gets or sets the latent downsampling factor. |
| `MaxDurationSeconds` | Gets or sets the maximum duration in seconds. |
| `MaxTextLength` | Gets or sets the maximum text sequence length. |
| `ModelSize` | Gets or sets the model size variant. |
| `NumInferenceSteps` | Gets or sets the number of diffusion steps. |
| `NumMelBins` | Gets or sets the number of mel spectrogram bins. |
| `OnnxOptions` | Gets or sets the ONNX execution options. |
| `SampleRate` | Gets or sets the output sample rate in Hz. |
| `Stereo` | Gets or sets whether to generate stereo audio. |
| `UNetPath` | Gets or sets the path to the U-Net denoiser ONNX model. |
| `VaePath` | Gets or sets the path to the VAE ONNX model. |
| `VocoderPath` | Gets or sets the path to the HiFi-GAN vocoder ONNX model. |
| `WindowSize` | Gets or sets the FFT window size. |

