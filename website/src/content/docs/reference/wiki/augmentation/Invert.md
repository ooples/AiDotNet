---
title: "Invert<T>"
description: "Inverts all pixel values in the image (creates a negative)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Inverts all pixel values in the image (creates a negative).

## How It Works

Each pixel value is replaced with (max - value), where max is 255 for uint8 images
or 1.0 for normalized images. This creates a photographic negative effect.

