---
title: "Ade20kDataLoader<T>"
description: "Loads the ADE20K semantic segmentation dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the ADE20K semantic segmentation dataset.

## How It Works

ADE20K expects:

Features are flattened image pixels Tensor[N, H * W * 3].
Labels are flattened segmentation mask Tensor[N, H * W] with class indices.

