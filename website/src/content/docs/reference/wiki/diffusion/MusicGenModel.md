---
title: "MusicGenModel<T>"
description: "MusicGen - Diffusion-based music generation model with advanced musical controls."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Audio`

MusicGen - Diffusion-based music generation model with advanced musical controls.

## For Beginners

This model generates music with precise control:

Example prompts:

- "Upbeat electronic dance music at 128 BPM" -> EDM track
- "Sad piano ballad in A minor" -> emotional piano piece
- "Funky bass groove with drums" -> funk rhythm section
- "Orchestral film score, epic and dramatic" -> cinematic music

Advanced controls:

- BPM: Set exact tempo (60-200 BPM typical)
- Key: Major/minor keys (C major, A minor, etc.)
- Instruments: Specify or exclude instruments
- Style: Jazz, rock, classical, electronic, etc.

## How It Works

MusicGenModel is a specialized diffusion model for music generation that provides
fine-grained control over musical characteristics including:

1. Text-to-Music: Generate music from natural language descriptions
2. Melody Conditioning: Guide generation with a reference melody
3. Rhythm/Beat Conditioning: Generate music following a specific rhythm pattern
4. Tempo Control: Generate at specific BPM (beats per minute)
5. Key/Scale Guidance: Influence the musical key of generated content
6. Style Transfer: Transform existing music to different styles

Technical specifications:

- Sample rate: 32 kHz (high-quality music)
- Latent channels: 16 (more capacity for musical structure)
- Mel channels: 128
- Duration: Up to 60 seconds
- Guidance scale: 3.0-7.0 typical

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MusicGenModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,AudioVAE<>,IConditioningModule<>,MusicGenSize,Int32,Double,Nullable<Int32>)` | Initializes a new MusicGen model with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `MelodyEncoder` | Gets the melody encoder for melody conditioning. |
| `ModelSize` | Gets the model size variant. |
| `MusicVAE` | Gets the music VAE for direct access. |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `RhythmEncoder` | Gets the rhythm encoder for beat conditioning. |
| `SupportsAudioToAudio` |  |
| `SupportsTextToAudio` |  |
| `SupportsTextToMusic` |  |
| `SupportsTextToSpeech` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BlendAudio(Tensor<>,Tensor<>,Int32)` | Blends two audio tensors with crossfade. |
| `BlendConditions(Tensor<>,Tensor<>,Double)` | Blends two conditioning tensors. |
| `Clone` |  |
| `CombineMusicConditions(Tensor<>,Tensor<>,Tensor<>)` | Combines multiple music conditioning sources. |
| `ContinueMusic(Tensor<>,String,Double,Double,Int32,Double,Nullable<Int32>)` | Generates music continuation from an audio prompt. |
| `CreateTempoEmbedding(Nullable<Int32>,Int32[])` | Creates tempo embedding for BPM conditioning. |
| `DeepCopy` |  |
| `GenerateContinuationLatent(Tensor<>,Tensor<>,String,Double,Int32,Double,Nullable<Int32>)` | Generates continuation latent conditioned on prompt. |
| `GenerateFromMelody(Tensor<>,String,Double,String,Int32,Double,Nullable<Int32>)` | Generates music conditioned on a reference melody. |
| `GenerateFromRhythm(Tensor<>,String,Double,String,Int32,Double,Nullable<Int32>)` | Generates music conditioned on a rhythm/beat pattern. |
| `GenerateMusic(String,String,Nullable<Double>,Int32,Double,Nullable<Int32>)` | Generates music from a text prompt. |
| `GenerateMusicLatent(String,String,Double,Int32,Double,Tensor<>,Tensor<>,Nullable<Int32>,Nullable<Int32>,Double)` | Core generation method with all conditioning options. |
| `GenerateMusicWithTempo(String,Int32,String,Nullable<Double>,Int32,Double,Nullable<Int32>)` | Generates music with specific tempo (BPM) control. |
| `GetParameters` |  |
| `InitializeLayers(UNetNoisePredictor<>,AudioVAE<>,MusicGenSize,Nullable<Int32>)` | Initializes the model layers, using provided components or creating defaults. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DEFAULT_BPM` | Default BPM for music generation. |
| `MUSICGEN_BASE_CHANNELS` | MusicGen U-Net base channels. |
| `MUSICGEN_CONTEXT_DIM` | Context dimension for conditioning. |
| `MUSICGEN_LATENT_CHANNELS` | MusicGen latent space channels (larger for musical structure). |
| `MUSICGEN_MAX_DURATION` | Maximum supported duration in seconds. |
| `MUSICGEN_MEL_CHANNELS` | MusicGen mel spectrogram channels. |
| `MUSICGEN_SAMPLE_RATE` | MusicGen default sample rate for high-quality music. |
| `_melodyEncoder` | Melody encoder for melody conditioning. |
| `_modelSize` | Model size variant. |
| `_musicVAE` | The AudioVAE for high-resolution music encoding/decoding. |
| `_rhythmEncoder` | Rhythm encoder for beat conditioning. |
| `_textConditioner` | Primary text conditioning module. |
| `_unet` | The U-Net noise predictor optimized for music. |

