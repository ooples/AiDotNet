---
title: "AugmentationConfig"
description: "Unified configuration for data augmentation with industry-standard defaults."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Augmentation`

Unified configuration for data augmentation with industry-standard defaults.

## For Beginners

Data augmentation creates variations of your training data
(like flipping images, adding noise, or shuffling words) to help your model learn better
and become more robust. This configuration controls how augmentation works.

## How It Works

**Key features:**

- Automatic data-type detection (image, tabular, audio, text, video)
- Industry-standard defaults that work well out-of-the-box
- Test-Time Augmentation (TTA) enabled by default for better predictions
- Modality-specific settings for fine-tuning

**Example - Simple usage with defaults:**

**Example - Custom configuration:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AugmentationConfig` | Creates a new augmentation configuration with industry-standard defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AudioSettings` | Gets or sets audio-specific augmentation settings. |
| `CustomAugmenter` | User-supplied custom augmenter object — when set, BuildAsync will invoke its `Apply` method on each training input before the optimizer runs. |
| `EnableTTA` | Gets or sets whether Test-Time Augmentation is enabled during inference. |
| `ImageSettings` | Gets or sets image-specific augmentation settings. |
| `IsEnabled` | Gets or sets whether augmentation is enabled. |
| `Probability` | Gets or sets the global probability of applying any augmentation. |
| `Seed` | Gets or sets the random seed for reproducible augmentations. |
| `TTAAggregation` | Gets or sets how to aggregate TTA predictions. |
| `TTAIncludeOriginal` | Gets or sets whether to include the original (non-augmented) sample in TTA. |
| `TTANumAugmentations` | Gets or sets the number of augmented samples for TTA. |
| `TabularSettings` | Gets or sets tabular-specific augmentation settings. |
| `TextSettings` | Gets or sets text-specific augmentation settings. |
| `VideoSettings` | Gets or sets video-specific augmentation settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForAudio` | Creates an augmentation configuration for audio data with standard defaults. |
| `ForImages` | Creates an augmentation configuration for image data with standard defaults. |
| `ForTabular` | Creates an augmentation configuration for tabular data with standard defaults. |
| `ForText` | Creates an augmentation configuration for text data with standard defaults. |
| `ForVideo` | Creates an augmentation configuration for video data with standard defaults. |
| `GetConfiguration` | Gets the configuration as a dictionary for logging or serialization. |
| `SetCustomAugmenter(IAugmentation<,>)` | Strongly-typed setter that constrains the augmenter's type arguments at the call site. |

