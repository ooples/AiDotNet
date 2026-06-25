---
title: "DiceLossFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Dice Loss to evaluate model performance for image segmentation and other tasks where overlap between predictions and actual values is important."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Dice Loss to evaluate model performance for image segmentation and other tasks where overlap between predictions and actual values is important.

## For Beginners

This calculator helps you evaluate how well your model is performing on tasks where you need to 
identify specific regions or areas in data, especially in images.

Dice Loss (named after Lee R. Dice who created the Dice coefficient) measures the overlap between 
two sets - in this case, between your model's predictions and the actual correct regions.

Think of it like comparing two traced outlines:

- If both outlines perfectly match, the Dice score is 1 (and the loss is 0)
- If they partially overlap, the score is between 0 and 1 (and the loss is between 1 and 0)
- If they don't overlap at all, the score is 0 (and the loss is 1)

Some common applications include:

- Medical image segmentation (identifying organs, tumors, or other structures in medical scans)
- Satellite image analysis (identifying buildings, roads, or forests in aerial images)
- Object detection in photos (identifying the exact boundaries of objects)
- Document layout analysis (identifying paragraphs, images, or tables in documents)

Dice Loss is particularly useful when dealing with imbalanced data, such as when the region you're 
trying to identify is much smaller than the background (like finding a small tumor in a large scan).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiceLossFitnessCalculator(DataSetType)` | Initializes a new instance of the DiceLossFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Dice Loss between predicted and actual values. |

