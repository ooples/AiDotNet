---
title: "AudioMAE<T>"
description: "Audio-MAE (Masked Autoencoders for Audio) model for audio classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Classification`

Audio-MAE (Masked Autoencoders for Audio) model for audio classification.

## For Beginners

Audio-MAE learns by hiding 80% of a spectrogram and trying to
reconstruct it. This "fill in the blanks" approach forces the model to deeply understand
audio patterns. After pre-training, only the encoder is kept for classification.

**Usage:**

## How It Works

Audio-MAE (Xu et al., NeurIPS 2022) applies Masked Autoencoders to audio spectrograms.
By masking 80% of spectrogram patches and reconstructing them, Audio-MAE learns rich
audio representations achieving 47.3% mAP on AudioSet and 97.0% on ESC-50.

**References:**

- Paper: "Masked Autoencoders that Listen" (Xu et al., NeurIPS 2022)
- Repository: https://github.com/facebookresearch/AudioMAE

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioMAE(NeuralNetworkArchitecture<>,AudioMAEOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an Audio-MAE model for native training mode. |
| `AudioMAE(NeuralNetworkArchitecture<>,String,AudioMAEOptions)` | Creates an Audio-MAE model for ONNX inference mode. |

