---
title: "CelebADataLoader<T>"
description: "Loads the CelebA face attributes dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the CelebA face attributes dataset.

## How It Works

CelebA expects:

Features are flattened image pixels Tensor[N, H * W * 3].
Labels are binary attributes Tensor[N, 40].

