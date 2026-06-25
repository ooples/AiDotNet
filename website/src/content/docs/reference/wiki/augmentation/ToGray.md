---
title: "ToGray<T>"
description: "Converts to grayscale with random channel weights, outputting 3 channels."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Converts to grayscale with random channel weights, outputting 3 channels.

## How It Works

Unlike RgbToGrayscale which uses fixed weights and can output 1 channel, ToGray
uses random weights for the RGB channels and always outputs 3 channels (grayscale
replicated). This adds stochastic color invariance as augmentation.

