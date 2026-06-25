---
title: "ImageSafetyConfig"
description: "Configuration for image safety modules."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Safety`

Configuration for image safety modules.

## For Beginners

These settings control how images are checked for harmful content.
NSFW detection catches adult content, violence detection catches graphic imagery,
and deepfake detection identifies AI-manipulated faces.

## How It Works

**References:**

- UnsafeBench: GPT-4V achieves top F1 across 11 categories (Qu et al., 2024)
- Vision Transformers outperform CNNs for sensitive image classification (2024)

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassifierType` | Gets or sets the type of image safety classifier to use. |
| `DeepfakeDetection` | Gets or sets whether deepfake/AI-generated image detection is enabled. |
| `NSFWDetection` | Gets or sets whether NSFW (sexual content) detection is enabled. |
| `NSFWThreshold` | Gets or sets the NSFW detection threshold (0-1). |
| `ViolenceDetection` | Gets or sets whether violence detection is enabled. |
| `ViolenceThreshold` | Gets or sets the violence detection threshold (0-1). |

