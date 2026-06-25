---
title: "WavLMSEROptions"
description: "Configuration options for the WavLM-SER speech emotion recognition model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Emotion`

Configuration options for the WavLM-SER speech emotion recognition model.

## For Beginners

WavLM-SER takes a model that first learned to understand speech in
general, then teaches it to recognize emotions specifically. Because WavLM already understands
the deep patterns of human speech, it picks up on subtle emotional cues that simpler models
miss—like the slight tremor in a fearful voice or the rise in pitch when someone is excited.

## How It Works

WavLM-SER fine-tunes the WavLM self-supervised model (Chen et al., 2022) for speech emotion
recognition. WavLM's pre-training on masked speech prediction and denoising produces robust
features that, when fine-tuned for SER, achieve state-of-the-art results on IEMOCAP
(weighted accuracy 71%+) and are robust to noise and recording conditions.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmotionLabels` | Gets or sets the emotion label names. |
| `FeatureEncoderDim` | Gets or sets the CNN feature encoder output dimension. |
| `FeatureLayerIndex` | Gets or sets which WavLM layer to use for features (-1 = weighted sum). |
| `FeedForwardDim` | Gets or sets the feed-forward hidden dimension. |
| `FftSize` | Gets or sets the FFT window size. |
| `HiddenDim` | Gets or sets the Transformer hidden dimension. |
| `HopLength` | Gets or sets the hop length between frames. |
| `IncludeArousalValence` | Gets or sets whether to include arousal/valence estimation. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumClasses` | Gets or sets the number of emotion classes. |
| `NumLayers` | Gets or sets the number of Transformer layers. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `Variant` | Gets or sets the model variant ("base" or "large"). |

