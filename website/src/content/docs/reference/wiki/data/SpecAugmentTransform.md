---
title: "SpecAugmentTransform<T>"
description: "Applies SpecAugment data augmentation (Park et al., 2019) to spectrogram tensors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Transforms`

Applies SpecAugment data augmentation (Park et al., 2019) to spectrogram tensors.
Performs time masking and frequency masking to improve model robustness.

## For Beginners

SpecAugment is a simple but effective augmentation for audio models.
It randomly "erases" parts of the spectrogram during training to prevent overfitting.

## How It Works

SpecAugment applies two types of masking to spectrograms:

- **Frequency masking**: Masks consecutive frequency bins (vertical stripes).
- **Time masking**: Masks consecutive time frames (horizontal stripes).

Reference: Park, D.S. et al. "SpecAugment: A Simple Data Augmentation Method for ASR." Interspeech 2019.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpecAugmentTransform(Int32,Int32,Int32,Int32,Nullable<Int32>)` | Creates a new SpecAugment transform. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Tensor<>)` | Applies SpecAugment masking to a spectrogram tensor. |

