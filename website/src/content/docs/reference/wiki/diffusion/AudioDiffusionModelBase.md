---
title: "AudioDiffusionModelBase<T>"
description: "Base class for audio diffusion models that generate sound and music."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Diffusion`

Base class for audio diffusion models that generate sound and music.

## For Beginners

This is the foundation for audio generation models like AudioLDM.
It extends latent diffusion to work with audio by converting sound to spectrograms
(visual representations of sound) and back.

## How It Works

This abstract base class provides common functionality for all audio diffusion models,
including text-to-audio generation, text-to-music, text-to-speech, and audio transformation.

How audio diffusion works:

1. Audio is converted to a mel spectrogram (frequency vs time image)
2. The spectrogram is encoded to latent space (like images)
3. Diffusion denoising happens in latent space
4. The result is decoded to a spectrogram
5. A vocoder converts the spectrogram back to audio

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioDiffusionModelBase(DiffusionModelOptions<>,INoiseScheduler<>,Int32,Double,Int32,NeuralNetworkArchitecture<>)` | Initializes a new instance of the AudioDiffusionModelBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultDurationSeconds` |  |
| `FFTSize` | Gets the FFT window size. |
| `HopLength` | Gets the hop length for spectrogram computation. |
| `MaxFrequency` | Gets the maximum frequency for mel filterbank. |
| `MelChannels` |  |
| `MinFrequency` | Gets the minimum frequency for mel filterbank. |
| `SampleRate` |  |
| `SupportsAudioToAudio` |  |
| `SupportsTextToAudio` |  |
| `SupportsTextToMusic` |  |
| `SupportsTextToSpeech` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AudioToAudio(Tensor<>,String,String,Double,Int32,Double,Nullable<Int32>)` |  |
| `CombineTextAndSpeakerEmbeddings(Tensor<>,Tensor<>)` | Combines text embedding with speaker embedding for TTS. |
| `ConcatenateAudio(Tensor<>,Tensor<>)` | Concatenates two audio waveforms. |
| `ConcatenateLatents(Tensor<>,Tensor<>)` | Concatenates two latent tensors along the time dimension. |
| `ContinueAudio(Tensor<>,String,Double,Int32,Nullable<Int32>)` |  |
| `CopyAudioToBatch(Span<>,Tensor<>,Int32,Int32)` | Copies audio samples to a specific batch position. |
| `CopyMelToBatch(Span<>,Tensor<>,Int32,Int32,Int32)` | Copies a mel spectrogram to a specific batch position. |
| `EnsureGriffinLimProcessorInitialized` | Ensures the Griffin-Lim processor is initialized. |
| `EnsureMelSpectrogramProcessorInitialized` | Ensures the mel spectrogram processor is initialized. |
| `EstimateSpeechDuration(String,Double)` | Estimates speech duration based on text length and speaking rate. |
| `ExtractBatchMel(Tensor<>,Int32,Int32,Int32)` | Extracts a single mel spectrogram from a batch tensor. |
| `ExtractBatchWaveform(Tensor<>,Int32,Int32)` | Extracts a single waveform from a batch tensor. |
| `ExtractLatentContext(Tensor<>)` | Extracts context from the end of latent representation for continuation. |
| `ExtractSpeakerEmbedding(Tensor<>)` |  |
| `GenerateFromText(String,String,Nullable<Double>,Int32,Double,Nullable<Int32>)` |  |
| `GenerateMusic(String,String,Nullable<Double>,Int32,Double,Nullable<Int32>)` |  |
| `MelSpectrogramToWaveform(Tensor<>)` |  |
| `PredictNoiseWithContext(Tensor<>,Int32,Tensor<>,Tensor<>)` | Predicts noise with context from previous audio. |
| `TextToSpeech(String,Tensor<>,Double,Int32,Nullable<Int32>)` |  |
| `WaveformToMelSpectrogram(Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_defaultDurationSeconds` | Default audio duration in seconds. |
| `_griffinLimProcessor` | GPU-accelerated Griffin-Lim audio reconstructor. |
| `_melChannels` | Number of mel spectrogram channels. |
| `_melSpectrogramProcessor` | GPU-accelerated mel spectrogram processor. |
| `_sampleRate` | Sample rate in Hz. |

