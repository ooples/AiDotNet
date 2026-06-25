---
title: "AVHuBERT<T>"
description: "AV-HuBERT: audio-visual speech recognition with HuBERT"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Robust`

AV-HuBERT: audio-visual speech recognition with HuBERT

## For Beginners

AV-HuBERT extends HuBERT to jointly learn from audio and visual (lip) modalities. The model uses separate audio (CNN) and visual (ResNet) feature encoders followed by a shared Transformer that learns multimodal representations. During pre-training...

## How It Works

**References:**

- Paper: "Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction" (Shi et al., Meta, 2022)

AV-HuBERT extends HuBERT to jointly learn from audio and visual (lip) modalities. The model uses separate audio (CNN) and visual (ResNet) feature encoders followed by a shared Transformer that learns multimodal representations. During pre-training, both audio and visual streams are masked, and the model predicts cluster assignments from the combined input. For audio-only ASR, AV-HuBERT leverages pre-trained representations that capture cross-modal speech dynamics.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using AV-HuBERT's multimodal pre-trained encoder with CTC. |

