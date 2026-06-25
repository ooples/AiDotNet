---
title: "IAugmentationRecommender"
description: "Interface for recommending augmentations based on task and data characteristics."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Augmentation`

Interface for recommending augmentations based on task and data characteristics.

## For Beginners

Different tasks need different augmentations.
Object detection needs augmentations that preserve bounding boxes,
while pose estimation needs ones that correctly transform keypoints.
This recommender helps choose the right augmentations automatically.

## How It Works

This interface enables integration with agent systems and AutoML pipelines
by providing intelligent augmentation recommendations based on:

- The type of ML task being performed
- Characteristics of the dataset
- Best practices from research and industry

## Methods

| Method | Summary |
|:-----|:--------|
| `GetDefaultPolicy(AugmentationTaskType,Double)` | Gets a pre-configured augmentation policy for a task. |
| `GetRecommendations(AugmentationTaskType,DatasetCharacteristics)` | Gets recommendations for augmentations based on task and data. |
| `ValidateAugmentations(IEnumerable<String>,AugmentationTaskType)` | Validates whether augmentations are compatible with the task. |

