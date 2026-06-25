---
title: "IPanopticSegmentation<T>"
description: "Interface for panoptic segmentation models that unify semantic and instance segmentation."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for panoptic segmentation models that unify semantic and instance segmentation.

## For Beginners

Panoptic segmentation gives you the most complete picture.

For a street scene, you get:

- "Stuff" regions: road (no instance), sky (no instance), building (no instance)
- "Things" instances: person #1, person #2, car #1, car #2, bicycle #1

This is useful when you need to know both "what is everywhere" AND "how many of each thing".

Models implementing this interface:

- Mask2Former (CVPR 2022, 57.8 PQ on COCO)
- kMaX-DeepLab (CVPR 2023, cross-attention as clustering)
- ODISE (CVPR 2023, diffusion features)
- OneFormer (CVPR 2023, text-conditioned)

## How It Works

Panoptic segmentation provides a complete scene understanding by combining:

- Semantic segmentation for "stuff" classes (sky, road, grass — amorphous regions)
- Instance segmentation for "things" classes (car, person, dog — countable objects)

Every pixel receives both a class label and an instance ID (for thing classes).

## Properties

| Property | Summary |
|:-----|:--------|
| `NumStuffClasses` | Gets the number of "stuff" classes (amorphous regions like sky, road). |
| `NumThingClasses` | Gets the number of "thing" classes (countable objects like car, person). |

## Methods

| Method | Summary |
|:-----|:--------|
| `SegmentPanoptic(Tensor<>)` | Performs panoptic segmentation on an image. |

