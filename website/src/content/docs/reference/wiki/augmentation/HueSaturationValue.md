---
title: "HueSaturationValue<T>"
description: "Randomly adjusts hue, saturation, and value in HSV space."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Randomly adjusts hue, saturation, and value in HSV space.

## How It Works

Converts the image to HSV, applies random shifts to each component, then converts
back to RGB. This provides intuitive color augmentation by directly manipulating color
properties.

