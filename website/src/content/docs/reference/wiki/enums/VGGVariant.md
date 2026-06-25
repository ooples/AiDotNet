---
title: "VGGVariant"
description: "Defines the available VGG network architecture variants."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the available VGG network architecture variants.

## For Beginners

VGG networks are named after the Visual Geometry Group that created them.
The number in the name (e.g., VGG16) refers to the total number of weight layers in the network.
For example, VGG16 has 13 convolutional layers and 3 fully connected layers, totaling 16 weight layers.
These networks were groundbreaking because they showed that network depth is critical for good performance.
Despite being older architectures, they remain popular for transfer learning and as baselines.

## How It Works

VGG networks are a family of deep convolutional neural networks developed by the Visual Geometry Group
at Oxford University. They are characterized by their use of small (3x3) convolution filters stacked
in increasing depth, which allows them to learn complex features while keeping the number of parameters
manageable.

**Batch Normalization Variants:** The "_BN" suffix indicates variants that include batch normalization
layers after each convolutional layer. Batch normalization helps stabilize training and often allows
for faster convergence and better final accuracy.

## Fields

| Field | Summary |
|:-----|:--------|
| `VGG11` | VGG-11: 11 weight layers (8 conv + 3 FC). |
| `VGG11_BN` | VGG-11 with batch normalization after each convolutional layer. |
| `VGG13` | VGG-13: 13 weight layers (10 conv + 3 FC). |
| `VGG13_BN` | VGG-13 with batch normalization after each convolutional layer. |
| `VGG16` | VGG-16: 16 weight layers (13 conv + 3 FC). |
| `VGG16_BN` | VGG-16 with batch normalization after each convolutional layer. |
| `VGG19` | VGG-19: 19 weight layers (16 conv + 3 FC). |
| `VGG19_BN` | VGG-19 with batch normalization after each convolutional layer. |

