---
title: "TrainTestSplitter<T>"
description: "Simple random train/test splitter that divides data into two sets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Basic`

Simple random train/test splitter that divides data into two sets.

## For Beginners

This is the simplest and most common way to split your data.
It randomly divides your dataset into:

- Training set: Data your model learns from
- Test set: Data used to evaluate how well your model performs on unseen data

## How It Works

**Industry Standard:** An 80/20 split (80% train, 20% test) is very common.
For smaller datasets, you might use 70/30 to get more test samples.

**When to Use:**

- Large datasets (10,000+ samples)
- Quick experiments
- When you don't need hyperparameter tuning (otherwise use train/val/test)

**Example Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TrainTestSplitter(Double,Boolean,Int32)` | Creates a new train/test splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |
| `SplitIndicesOnly(Int32,Vector<>)` |  |

