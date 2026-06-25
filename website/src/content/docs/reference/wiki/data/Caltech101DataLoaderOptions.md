---
title: "Caltech101DataLoaderOptions"
description: "Configuration options for the Caltech-101 image classification data loader (Fei-Fei et al."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the Caltech-101 image classification data loader (Fei-Fei et al. 2004).

## How It Works

Caltech-101 contains ≈ 9,000 images across 101 object categories plus a
background "BACKGROUND_Google" class (102 total). Image counts per
category vary from 40 to 800. Pre-CNN era benchmark, still used for
few-shot studies. Standard practice samples ≤ 30 images/class for
training and uses the rest for testing.

## Properties

| Property | Summary |
|:-----|:--------|
| `TrainImagesPerClass` | Per-class images for the training split. |

