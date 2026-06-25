---
title: "PANNs<T>"
description: "PANNs (Pre-trained Audio Neural Networks) CNN14 model for audio classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Classification`

PANNs (Pre-trained Audio Neural Networks) CNN14 model for audio classification.

## For Beginners

PANNs is like having increasingly specialized detectors stacked on top
of each other. The first layers detect simple patterns (edges, tones), middle layers detect
intermediate patterns (harmonics, rhythms), and final layers detect complete sounds (speech,
music, dog bark). Being CNN-based, it is faster than Transformer models but slightly less accurate.

**Usage:**

## How It Works

PANNs (Kong et al., IEEE/ACM TASLP 2020) provides pre-trained CNN-based audio classification
models. The flagship CNN14 achieves 43.1% mAP on AudioSet-2M and is widely used as a feature
extractor for downstream audio tasks.

**Architecture:** CNN14 is a 14-layer convolutional neural network:

- **Input**: 64-bin log-mel spectrogram at 32 kHz
- **6 CNN blocks**: Each with two 3x3 conv layers, batch norm, ReLU, and 2x2 avg pooling
- **Channel progression**: 64 -> 128 -> 256 -> 512 -> 1024 -> 2048
- **Global pooling**: Average and max pooling combined
- **Classification head**: 2048-dim FC -> 527-class sigmoid output

**References:**

- Paper: "PANNs: Large-Scale Pretrained Audio Neural Networks" (Kong et al., 2020)
- Repository: https://github.com/qiuqiangkong/audioset_tagging_cnn

