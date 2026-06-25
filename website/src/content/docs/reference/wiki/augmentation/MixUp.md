---
title: "MixUp<T>"
description: "Blends two images together by weighted averaging (MixUp augmentation)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Blends two images together by weighted averaging (MixUp augmentation).

## For Beginners

Imagine blending two photos together like a double exposure.
If you blend a photo of a cat with a photo of a dog, you get something that's partly
cat and partly dog. The label becomes a mix too: maybe 70% cat and 30% dog.
This teaches the model that the world isn't black-and-white, and improves generalization.

## How It Works

MixUp is a powerful regularization technique that creates virtual training examples by
taking linear combinations of two training samples and their labels. Given two images
(x₁, y₁) and (x₂, y₂), MixUp creates a new training sample:
x' = λx₁ + (1-λ)x₂
y' = λy₁ + (1-λ)y₂
where λ is sampled from a Beta distribution.

**When to use:**

- Image classification with sufficient training data
- When you want smoother decision boundaries
- As a regularization alternative or complement to dropout

**When NOT to use:**

- Object detection (bounding boxes can't be meaningfully mixed)
- Semantic segmentation (mixed masks don't make sense)
- When you need hard labels for downstream processing

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MixUp(Double,Double)` | Creates a new MixUp augmentation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies MixUp augmentation (single image version - requires external pairing). |
| `ApplyMixUp(ImageTensor<>,ImageTensor<>,Vector<>,Vector<>,AugmentationContext<>)` | Applies MixUp to blend two images together. |
| `GetParameters` |  |

