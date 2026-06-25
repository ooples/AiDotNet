---
title: "IAudioLanguageModel<T>"
description: "Interface for multimodal audio-language models that understand and reason about audio."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for multimodal audio-language models that understand and reason about audio.

## For Beginners

Audio-language models are like ChatGPT but for audio. You can
play them a sound and ask questions like "What instruments are playing?" or "Describe
the audio scene." They combine the ability to hear (audio encoder) with the ability
to understand and respond in natural language (language model).

How they work:

1. Audio encoder converts sound to features
2. An adapter aligns audio features with the language model
3. The language model processes both audio features and text prompt
4. It generates a text response about the audio

Common use cases:

- Audio captioning: "Describe this sound" -> "A bird singing in a forest"
- Audio QA: "What instrument is playing?" -> "A piano"
- Audio scene understanding: "Where was this recorded?" -> "An indoor concert hall"
- Audio reasoning: "Is this recording happy or sad?" -> "Happy, upbeat tone"

## How It Works

Audio-language models combine audio understanding with natural language processing,
enabling tasks like audio captioning, audio question answering, and audio-guided
reasoning. They take audio input and text prompts and produce text responses.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxAudioDurationSeconds` | Gets the maximum audio duration in seconds that the model can process. |
| `MaxResponseTokens` | Gets the maximum number of tokens the model can generate in a response. |
| `SampleRate` | Gets the sample rate expected by the audio encoder. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Caption(Tensor<>,Int32)` | Generates a caption describing the audio content. |
| `ExtractAudioEmbeddings(Tensor<>)` | Extracts audio embeddings that can be used for downstream tasks. |
| `GetCapabilities` | Gets the list of capabilities this model supports. |
| `Understand(Tensor<>,String,Int32,Double)` | Generates a text response about the given audio based on a text prompt. |
| `UnderstandAsync(Tensor<>,String,Int32,Double,CancellationToken)` | Generates a text response about audio asynchronously. |

