---
title: "HuBERTSEROptions"
description: "Configuration options for the HuBERT-based Speech Emotion Recognition model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Emotion`

Configuration options for the HuBERT-based Speech Emotion Recognition model.

## For Beginners

HuBERT-SER uses a pre-trained speech understanding model (HuBERT)
and teaches it to recognize emotions. HuBERT first learns general speech patterns from
millions of hours of audio, then is specialized for emotion detection.

## How It Works

HuBERT-SER fine-tunes the HuBERT (Hsu et al., 2021) self-supervised model for speech
emotion recognition. HuBERT learns speech representations through masked prediction,
and when fine-tuned for SER, achieves strong results on IEMOCAP (69.7% weighted accuracy).

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassifierHiddenDim` | Gets or sets the classification head hidden dimension. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmotionLabels` | Gets or sets the emotion labels. |
| `FeedForwardDim` | Gets or sets the feed-forward dimension. |
| `FftSize` | Gets or sets the FFT window size in samples. |
| `HopLength` | Gets or sets the hop length between frames in samples. |
| `IncludeArousalValence` | Gets or sets whether to include arousal/valence regression heads. |
| `LearningRate` | Gets or sets the learning rate for training. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumClasses` | Gets or sets the number of emotion classes. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `NumTransformerLayers` | Gets or sets the number of Transformer layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `TransformerDim` | Gets or sets the Transformer encoder dimension. |
| `Variant` | Gets or sets the model variant (base or large). |
| `WeightDecay` | Gets or sets the weight decay for regularization. |

