---
title: "SceneGraphSafetyClassifier<T>"
description: "Scene graph-based image safety classifier that analyzes spatial relationships between detected entities to identify unsafe content configurations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Image`

Scene graph-based image safety classifier that analyzes spatial relationships between
detected entities to identify unsafe content configurations.

## For Beginners

This classifier works like a detective examining a scene: first it
identifies what objects/regions are present (skin, dark areas, bright areas), then checks
how they relate to each other spatially. For example, a weapon-shaped object near a
person-shaped region is more concerning than either alone.

## How It Works

Rather than classifying the image as a whole, this classifier segments the image into
regions, characterizes each region's visual properties, and then analyzes spatial
relationships between regions. Certain spatial configurations (e.g., skin-colored regions
in specific arrangements, weapon-shaped objects near person-shaped regions) indicate
unsafe content that whole-image classifiers may miss.

**References:**

- USD: Scene-graph-based NSFW detection for text-to-image (USENIX Security 2025)
- Scene Graph Generation survey: methods, challenges, applications (2024)
- OmniSafeBench-MM: 9 risk domains with 50 fine-grained categories (2025, arxiv:2512.06589)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SceneGraphSafetyClassifier(Double,Int32)` | Initializes a new scene graph safety classifier. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateImage(Tensor<>)` |  |

