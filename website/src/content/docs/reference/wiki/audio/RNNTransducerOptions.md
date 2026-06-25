---
title: "RNNTransducerOptions"
description: "Configuration options for the RNN-Transducer (RNN-T) speech recognition model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.SpeechRecognition`

Configuration options for the RNN-Transducer (RNN-T) speech recognition model.

## For Beginners

RNN-T is a real-time speech recognizer ideal for live transcription.
Unlike batch models (like Whisper), it processes audio as it arrives, making it perfect for
live captioning and voice assistants. It has three parts: an encoder (listens), a predictor
(remembers what was said), and a joiner (combines both to output the next word).

## How It Works

RNN-Transducer (Graves, 2012; He et al., 2019) combines an audio encoder with a label
prediction network and a joint network to produce a streaming ASR system. Unlike CTC,
RNN-T can model output dependencies through its prediction network, achieving strong
results without an external language model.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDim` | Gets or sets the embedding dimension for output tokens. |
| `EncoderDim` | Gets or sets the encoder hidden dimension. |
| `JointDim` | Gets or sets the joint network hidden dimension. |
| `Language` | Gets or sets the language code. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumEncoderHeads` | Gets or sets the number of encoder attention heads. |
| `NumEncoderLayers` | Gets or sets the number of encoder layers. |
| `NumMels` | Gets or sets the number of mel spectrogram channels. |
| `NumPredictionLayers` | Gets or sets the number of prediction LSTM layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `PredictionDim` | Gets or sets the prediction network hidden dimension. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Variant` | Gets or sets the model variant ("small", "medium", "large"). |
| `VocabSize` | Gets or sets the vocabulary size (subword tokens). |
| `Vocabulary` | Gets or sets the CTC/RNN-T vocabulary (characters or BPE tokens). |

