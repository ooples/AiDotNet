---
title: "PoolingType"
description: "Defines different methods for pooling (downsampling) data in neural networks, particularly in convolutional neural networks."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines different methods for pooling (downsampling) data in neural networks, particularly in convolutional neural networks.

## For Beginners

Pooling is like summarizing information to make it more manageable. Imagine you have a large, 
detailed photograph and you want to create a smaller version that still captures the important features. 
Pooling does this for AI models by taking groups of numbers (like pixels) and combining them into single values.

This serves two important purposes:

1. It reduces the amount of data the model needs to process, making it faster and more efficient
2. It helps the model focus on important features regardless of their exact position (called "positional invariance")

For example, if a model is trying to recognize a cat in a photo, pooling helps it identify cat features 
whether the cat is in the center, corner, or any other position in the image.

## Fields

| Field | Summary |
|:-----|:--------|
| `Average` | Takes the average value from each group of values. |
| `Max` | Takes the maximum value from each group of values. |

