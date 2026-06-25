---
title: "AudioFlamingo2<T>"
description: "Audio Flamingo 2 multimodal audio-language model for audio understanding with interleaved inputs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Multimodal`

Audio Flamingo 2 multimodal audio-language model for audio understanding with interleaved inputs.

## For Beginners

Audio Flamingo 2 gives a language AI the ability to hear. It can
listen to audio recordings and answer questions about them, generate descriptions, or
reason about what's happening in the audio scene. It works by connecting a pre-trained
audio encoder to a language model using a special adapter layer.

**Usage:**

## How It Works

Audio Flamingo 2 (2024) extends the Flamingo architecture for audio understanding with
interleaved audio-text inputs. It uses a frozen audio encoder with perceiver-style
cross-attention to adapt a pre-trained LLM for audio captioning, QA, and reasoning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioFlamingo2(NeuralNetworkArchitecture<>,AudioFlamingo2Options,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an Audio Flamingo 2 model in native training mode. |
| `AudioFlamingo2(NeuralNetworkArchitecture<>,String,AudioFlamingo2Options)` | Creates an Audio Flamingo 2 model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxAudioDurationSeconds` |  |
| `MaxResponseTokens` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Caption(Tensor<>,Int32)` |  |
| `ExtractAudioEmbeddings(Tensor<>)` |  |
| `GetCapabilities` |  |
| `Understand(Tensor<>,String,Int32,Double)` |  |
| `UnderstandAsync(Tensor<>,String,Int32,Double,CancellationToken)` |  |

