---
title: "AudioLDM2Model<T>"
description: "AudioLDM 2 - Enhanced Audio Latent Diffusion Model with dual text encoders."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Audio`

AudioLDM 2 - Enhanced Audio Latent Diffusion Model with dual text encoders.

## For Beginners

AudioLDM 2 generates higher-quality audio than AudioLDM 1:

Example prompts:

- "A symphony orchestra playing a dramatic crescendo" -> orchestral music
- "Footsteps on gravel with birds chirping" -> detailed soundscape
- "Electric guitar riff with heavy distortion" -> rock music

The dual encoder architecture means:

- CLAP encoder understands audio concepts (instrument sounds, effects)
- T5/GPT-2 encoder understands language (descriptions, context)
- Combined, they produce audio that matches both sound and meaning

## How It Works

AudioLDM 2 is an improved version of AudioLDM with significant architectural enhancements
for better text-to-audio and text-to-music generation. Key improvements include:

1. Dual Text Encoders: Combines CLAP (audio-text) and T5/GPT-2 (language) embeddings
2. Larger Architecture: 384 base channels vs 256 in AudioLDM 1
3. Higher Resolution: 128 mel channels vs 64 for better audio quality
4. Improved Music Generation: Better temporal coherence and musical structure
5. Longer Duration Support: Up to 30 seconds of audio generation

Technical specifications:

- Sample rate: 16 kHz (speech/effects) or 48 kHz (high-quality music)
- Latent channels: 8
- Mel channels: 128 (double AudioLDM 1)
- Base channels: 384 (1.5x AudioLDM 1)
- Context dimension: 1024 (combined encoder output)
- Duration: Up to 30 seconds
- Guidance scale: 3.0-6.0 typical

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioLDM2Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,AudioVAE<>,IConditioningModule<>,IConditioningModule<>,AudioLDM2Variant,Int32,Double,Nullable<Int32>)` | Initializes a new AudioLDM 2 model with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AudioVAE` | Gets the AudioVAE for direct access. |
| `Conditioner` |  |
| `LanguageConditioner` | Gets the secondary language conditioning module. |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `SupportsAudioToAudio` |  |
| `SupportsTextToAudio` |  |
| `SupportsTextToMusic` |  |
| `SupportsTextToSpeech` |  |
| `VAE` |  |
| `Variant` | Gets the model variant. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddNoiseAtTimestep(Tensor<>,Tensor<>,Int32)` | Adds noise at a specific timestep for audio transformation. |
| `Clone` |  |
| `CombineEmbeddings(Tensor<>,Tensor<>)` | Combines embeddings from both encoders. |
| `ConcatenateEmbeddings(Tensor<>,Tensor<>)` | Concatenates CLAP and language embeddings along the feature dimension. |
| `DeepCopy` |  |
| `GenerateAudio(String,String,Nullable<Double>,Int32,Double,Nullable<Int32>)` | Generates audio from a text prompt using dual encoders. |
| `GenerateMusic(String,String,Nullable<Double>,Int32,Double,Nullable<Int32>)` | Generates music from a text prompt with enhanced musical understanding. |
| `GenerateVariations(Tensor<>,Int32,Double,Nullable<Int32>)` | Generates audio variations with enhanced diversity. |
| `GenerateWithDualEncoders(String,String,Double,Int32,Double,Nullable<Int32>)` | Generates latent using both encoders. |
| `GetParameters` |  |
| `InitializeLayers(UNetNoisePredictor<>,AudioVAE<>,AudioLDM2Variant,Nullable<Int32>)` | Initializes the U-Net, AudioVAE, and projection layers. |
| `InterpolateAudio(Tensor<>,Tensor<>,Int32)` | Interpolates between two audio samples in latent space. |
| `SetParameters(Vector<>)` |  |
| `TransformAudio(Tensor<>,String,String,Double,Int32,Double,Nullable<Int32>)` | Transforms audio based on a text prompt (audio-to-audio). |

## Fields

| Field | Summary |
|:-----|:--------|
| `AUDIOLDM2_BASE_CHANNELS` | AudioLDM 2 U-Net base channels (larger than AudioLDM 1). |
| `AUDIOLDM2_CONTEXT_DIM` | Combined context dimension from dual encoders. |
| `AUDIOLDM2_LATENT_CHANNELS` | AudioLDM 2 latent space channels. |
| `AUDIOLDM2_MAX_DURATION` | Maximum supported duration in seconds. |
| `AUDIOLDM2_MEL_CHANNELS` | AudioLDM 2 mel spectrogram channels (increased from 64 to 128). |
| `AUDIOLDM2_SAMPLE_RATE` | AudioLDM 2 default sample rate for high-quality audio. |
| `_audioVAE` | The AudioVAE for high-resolution mel spectrogram encoding/decoding. |
| `_clapConditioner` | Primary conditioning module (CLAP encoder for audio-text alignment). |
| `_languageConditioner` | Secondary conditioning module (T5/GPT-2 for language understanding). |
| `_projectionLayer` | Projection layer to combine dual encoder outputs. |
| `_unet` | The U-Net noise predictor optimized for AudioLDM 2. |
| `_variant` | Model variant configuration. |

