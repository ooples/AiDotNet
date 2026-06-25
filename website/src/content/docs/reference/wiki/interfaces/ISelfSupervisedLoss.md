---
title: "ISelfSupervisedLoss<T>"
description: "Interface for self-supervised loss functions used in meta-learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for self-supervised loss functions used in meta-learning.

## For Beginners

Self-supervised learning is like learning by creating your own practice problems.

Example: Rotation prediction for images

- Take an unlabeled image
- Rotate it by 0°, 90°, 180°, or 270°
- Train the model to predict which rotation was applied
- The model learns spatial relationships and features without needing class labels

This is powerful because:

1. You can use unlabeled data (which is often abundant)
2. The model learns useful features automatically
3. These features help with the actual task (classification, etc.)

Think of it like learning to recognize faces by first learning to identify if a photo is upside down.
You don't need to know who the person is to learn about facial features!

## How It Works

Self-supervised learning creates artificial tasks from unlabeled data, allowing models
to learn useful representations without explicit labels. This is particularly valuable
in meta-learning where the query set is often large but unlabeled.

**Common Self-Supervised Tasks:**

- **Rotation Prediction:** Predict rotation angle (0°, 90°, 180°, 270°)
- **Jigsaw Puzzles:** Solve scrambled image patches
- **Colorization:** Predict color from grayscale
- **Context Prediction:** Predict spatial relationships between patches
- **Contrastive Learning:** Learn to distinguish similar vs dissimilar examples

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateTask()` | Creates a self-supervised task from unlabeled input data. |

