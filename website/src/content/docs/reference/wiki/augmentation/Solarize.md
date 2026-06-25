---
title: "Solarize<T>"
description: "Inverts all pixel values above a threshold, creating a solarization effect."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Inverts all pixel values above a threshold, creating a solarization effect.

## For Beginners

Bright parts of the image get inverted (made dark) while dark
parts stay the same, creating a surreal effect. Used as augmentation in AutoAugment.

## How It Works

Solarization was originally a photographic darkroom effect. Pixels above the threshold
are inverted (value = max - value), while pixels below remain unchanged.

