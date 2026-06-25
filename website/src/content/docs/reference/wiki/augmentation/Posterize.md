---
title: "Posterize<T>"
description: "Reduces the number of bits per color channel, creating a poster-like effect."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Reduces the number of bits per color channel, creating a poster-like effect.

## For Beginners

Reducing from 8 bits (256 levels) to 4 bits (16 levels) per
channel makes the image look like a poster with fewer, more distinct colors.

## How It Works

Posterization quantizes pixel values to fewer discrete levels, creating flat areas
of color. This simulates low-bit-depth imaging and teaches robustness to quantization.

