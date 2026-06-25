---
title: "IDeepfakeDetector<T>"
description: "Interface for deepfake and AI-generated image detection modules."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Safety.Image`

Interface for deepfake and AI-generated image detection modules.

## For Beginners

A deepfake detector checks if an image is real or fake.
It looks for invisible signs of AI manipulation — patterns in the image frequencies,
inconsistencies in faces or backgrounds, and metadata clues that indicate the image
was generated or altered by AI.

## How It Works

Deepfake detectors analyze images for signs of AI generation or manipulation,
including frequency domain artifacts, facial/spatial inconsistencies, and
metadata/watermark provenance analysis.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetDeepfakeScore(Tensor<>)` | Gets the deepfake probability score for the given image (0.0 = authentic, 1.0 = fake). |

