---
title: "PredefinedSplitter<T>"
description: "User-specified split where indices for train/test are provided directly."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.GroupBased`

User-specified split where indices for train/test are provided directly.

## For Beginners

Sometimes you know exactly which samples should go where.
This splitter lets you specify the exact indices for training, testing,
and optionally validation.

## How It Works

**When to Use:**

- Reproducing a specific published split
- Domain-specific splitting requirements
- When automatic splitting isn't appropriate

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PredefinedSplitter(Int32[],Int32[],Int32[])` | Creates a new predefined splitter with specific indices. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `SupportsValidation` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |

