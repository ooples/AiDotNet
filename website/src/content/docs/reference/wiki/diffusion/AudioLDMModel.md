---
title: "AudioLDMModel<T>"
description: "Audio Latent Diffusion Model (AudioLDM) for text-to-audio generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Audio`

Audio Latent Diffusion Model (AudioLDM) for text-to-audio generation.

## For Beginners

AudioLDM lets you create sounds and music from text descriptions:

Example prompts:

- "A dog barking in a park" -> generates dog barking sounds
- "Rain falling on a window" -> generates rain sounds
- "Jazz piano playing softly" -> generates jazz piano music

How it works:

1. Text -> CLAP encoder -> text embedding (understands audio concepts)
2. Text embedding guides diffusion in latent space
3. Latent -> AudioVAE decoder -> mel spectrogram
4. Mel spectrogram -> Vocoder -> audio waveform

Key features:

- Text-to-audio: Generate sounds from descriptions
- Audio-to-audio: Transform sounds while preserving some characteristics
- Variable duration: Generate audio of different lengths
- Classifier-free guidance: Control how closely to follow the prompt

## How It Works

AudioLDM is a latent diffusion model specifically designed for audio generation.
It works by generating mel spectrograms in latent space and then converting
them to audio using a vocoder (like HiFi-GAN).

Technical specifications:

- Sample rate: 16 kHz (standard for speech/effects) or 48 kHz (music)
- Latent channels: 8
- Mel channels: 64 (AudioLDM) or 128 (AudioLDM 2)
- Duration: Typically 10 seconds, but configurable
- Guidance scale: 2.5-5.0 typical

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioLDMModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,AudioVAE<>,IConditioningModule<>,Int32,Double,Int32,Boolean,Nullable<Int32>)` | Initializes a new AudioLDM model with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AudioVAE` | Gets the AudioVAE used for encoding/decoding. |
| `Conditioner` |  |
| `IsVersion2` | Gets whether this is AudioLDM version 2. |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `SupportsAudioToAudio` |  |
| `SupportsTextToAudio` |  |
| `SupportsTextToMusic` |  |
| `SupportsTextToSpeech` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddNoiseAtTimestep(Tensor<>,Tensor<>,Int32)` | Adds noise at a specific timestep for audio-to-audio. |
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateAudio(String,String,Nullable<Double>,Int32,Double,Nullable<Int32>)` | Generates audio from a text prompt. |
| `GenerateMusic(String,String,Nullable<Double>,Int32,Double,Nullable<Int32>)` | Generates music from a text prompt. |
| `GenerateVariations(Tensor<>,Int32,Double,Nullable<Int32>)` | Generates audio variations from an input audio. |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |
| `TransformAudio(Tensor<>,String,String,Double,Int32,Double,Nullable<Int32>)` | Transforms audio based on a text prompt (audio-to-audio). |

## Fields

| Field | Summary |
|:-----|:--------|
| `AUDIOLDM_LATENT_CHANNELS` | Standard AudioLDM latent channels. |
| `AUDIOLDM_MEL_CHANNELS` | Standard AudioLDM mel channels. |
| `AUDIOLDM_SAMPLE_RATE` | Standard AudioLDM sample rate. |
| `_audioVAE` | The AudioVAE for mel spectrogram encoding/decoding. |
| `_conditioner` | The conditioning module (CLAP encoder). |
| `_isVersion2` | Whether this is AudioLDM 2 (larger model). |
| `_unet` | The U-Net noise predictor. |

