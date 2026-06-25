---
title: "Equalize<T>"
description: "Equalizes the image histogram per channel (same as HistogramEqualization with per-channel mode)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Equalizes the image histogram per channel (same as HistogramEqualization with per-channel mode).

## How It Works

This is a convenience wrapper matching the torchvision/AutoAugment Equalize operation.
Each color channel's histogram is equalized independently.

