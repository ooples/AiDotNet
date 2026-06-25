---
title: "GammaCorrection<T>"
description: "Applies gamma correction to adjust image brightness non-linearly."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Applies gamma correction to adjust image brightness non-linearly.

## For Beginners

Unlike brightness which adds/multiplies uniformly, gamma
correction affects dark and bright areas differently, giving more natural-looking
brightness adjustments.

## How It Works

Gamma correction raises each pixel value to the power of (1/gamma). Gamma > 1
brightens the image (especially dark areas), while gamma < 1 darkens it.

