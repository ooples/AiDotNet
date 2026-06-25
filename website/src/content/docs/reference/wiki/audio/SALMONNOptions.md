---
title: "SALMONNOptions"
description: "Configuration options for the SALMONN multimodal audio-language model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Multimodal`

Configuration options for the SALMONN multimodal audio-language model.

## For Beginners

SALMONN has two "ears": one specialized for speech (Whisper) and
one for general sounds (BEATs). This means it can understand both what people say AND
non-speech sounds like music, animal sounds, and environmental noise. It's like having
a translator who also happens to be an expert sound engineer.

## How It Works

SALMONN (Tang et al., 2024, Tsinghua/ByteDance) is a large language model with dual
audio encoders: a Whisper speech encoder and a BEATs audio encoder, connected to a
Vicuna LLM through a window-level Q-Former adapter. This dual-encoder design gives it
strong capability for both speech understanding and general audio understanding tasks.

## Properties

| Property | Summary |
|:-----|:--------|
| `AudioEncoderDim` | Gets or sets the audio encoder dimension (BEATs). |
| `DropoutRate` | Gets or sets the dropout rate. |
| `LMHiddenDim` | Gets or sets the language model hidden dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxAudioDurationSeconds` | Gets or sets the maximum audio duration in seconds. |
| `MaxResponseTokens` | Gets or sets the maximum response length in tokens. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAudioEncoderLayers` | Gets or sets the number of audio encoder layers. |
| `NumLMHeads` | Gets or sets the number of language model attention heads. |
| `NumLMLayers` | Gets or sets the number of language model layers. |
| `NumMels` | Gets or sets the number of mel spectrogram channels. |
| `NumQFormerLayers` | Gets or sets the number of Q-Former layers. |
| `NumQueryTokens` | Gets or sets the number of Q-Former query tokens. |
| `NumSpeechEncoderLayers` | Gets or sets the number of speech encoder layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `QFormerDim` | Gets or sets the Q-Former hidden dimension. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `SpeechEncoderDim` | Gets or sets the speech encoder dimension (Whisper). |
| `Temperature` | Gets or sets the sampling temperature. |
| `TopP` | Gets or sets the top-p (nucleus) sampling parameter. |
| `VocabSize` | Gets or sets the LM vocabulary size. |
| `WindowSize` | Gets or sets the window size for window-level Q-Former. |

