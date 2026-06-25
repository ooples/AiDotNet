---
title: "Emotion2VecOptions"
description: "Configuration options for the emotion2vec speech emotion recognition model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Emotion`

Configuration options for the emotion2vec speech emotion recognition model.

## For Beginners

emotion2vec is a model that "reads" emotions from speech.
It was trained on millions of speech samples to understand emotional patterns, then
fine-tuned to classify specific emotions like happy, sad, angry, etc.

## How It Works

emotion2vec (Ma et al., 2023) is a universal speech emotion representation model that
uses self-supervised pre-training on unlabeled speech data followed by fine-tuning.
It achieves state-of-the-art results across multiple SER benchmarks with a single model,
outperforming task-specific models.

## Properties

| Property | Summary |
|:-----|:--------|
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
| `WeightDecay` | Gets or sets the weight decay for regularization. |

