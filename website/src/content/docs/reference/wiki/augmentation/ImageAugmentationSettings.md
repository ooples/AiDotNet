---
title: "ImageAugmentationSettings"
description: "Image-specific augmentation settings with industry-standard defaults."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Augmentation`

Image-specific augmentation settings with industry-standard defaults.

## For Beginners

These settings control how images are transformed during
training. Defaults are based on best practices from Albumentations and torchvision.

## Properties

| Property | Summary |
|:-----|:--------|
| `BrightnessRange` | Gets or sets the brightness adjustment range. |
| `ContrastRange` | Gets or sets the contrast adjustment range. |
| `EnableColorJitter` | Gets or sets whether color jitter (brightness, contrast, saturation) is enabled. |
| `EnableCutMix` | Gets or sets whether CutMix is enabled. |
| `EnableCutout` | Gets or sets whether Cutout/CoarseDropout is enabled. |
| `EnableFlips` | Gets or sets whether horizontal flip is enabled. |
| `EnableGaussianBlur` | Gets or sets whether Gaussian blur is enabled. |
| `EnableGaussianNoise` | Gets or sets whether Gaussian noise is enabled. |
| `EnableMixUp` | Gets or sets whether MixUp is enabled. |
| `EnableRotation` | Gets or sets whether rotation is enabled. |
| `EnableVerticalFlip` | Gets or sets whether vertical flip is enabled. |
| `MixUpAlpha` | Gets or sets the MixUp alpha parameter. |
| `NoiseStdDev` | Gets or sets the standard deviation of Gaussian noise. |
| `RotationRange` | Gets or sets the rotation range in degrees. |
| `SaturationRange` | Gets or sets the saturation adjustment range. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetConfiguration` | Gets the configuration as a dictionary. |

