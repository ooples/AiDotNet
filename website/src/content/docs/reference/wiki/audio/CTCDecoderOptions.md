---
title: "CTCDecoderOptions"
description: "Configuration options for the CTC Decoder speech recognition model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.SpeechRecognition`

Configuration options for the CTC Decoder speech recognition model.

## For Beginners

CTC decoding solves a fundamental problem: audio frames don't
neatly align with letters or words. Some frames correspond to silence, some to the
middle of a vowel, and the model must figure out which frames correspond to which
characters. CTC introduces a "blank" token that the model outputs when nothing new
is being said, then the decoder collapses repeated characters and removes blanks.

Example of CTC decoding:
Raw output: h h - e e - l - l - l - o o
After collapse: h e l l o (where - is the blank token)

## How It Works

CTC (Connectionist Temporal Classification, Graves et al., 2006) is a training criterion
and decoding algorithm that allows sequence-to-sequence mapping without requiring exact
input-output alignment. With a greedy or beam-search decoder, CTC-trained models
directly output transcriptions from encoder features. CTC is used by wav2vec 2.0,
Conformer-CTC, DeepSpeech 2, and many production ASR systems.

## Properties

| Property | Summary |
|:-----|:--------|
| `BeamWidth` | Gets or sets the CTC beam width for beam search decoding. |
| `BlankTokenIndex` | Gets or sets the blank token index in the vocabulary. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderDim` | Gets or sets the encoder hidden dimension. |
| `FeedForwardDim` | Gets or sets the feed-forward dimension. |
| `LMWeight` | Gets or sets the language model weight for beam search. |
| `Language` | Gets or sets the default language code. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxAudioLengthSeconds` | Gets or sets the maximum audio length in seconds. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumEncoderLayers` | Gets or sets the number of encoder layers. |
| `NumMels` | Gets or sets the number of mel-spectrogram frequency bins. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `UseLM` | Gets or sets whether to use a language model for rescoring. |
| `Variant` | Gets or sets the model variant ("small", "medium", "large"). |
| `VocabSize` | Gets or sets the vocabulary size. |
| `Vocabulary` | Gets or sets the CTC vocabulary (characters or BPE tokens). |
| `WordInsertionPenalty` | Gets or sets the word insertion penalty for beam search. |

