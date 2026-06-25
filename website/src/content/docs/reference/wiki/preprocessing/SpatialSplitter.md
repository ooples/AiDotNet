---
title: "SpatialSplitter<T>"
description: "Spatial splitter for geographic or coordinate-based data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.DomainSpecific`

Spatial splitter for geographic or coordinate-based data.

## For Beginners

When working with geographic data (like weather stations,
property prices by location, or ecological surveys), spatial autocorrelation
means nearby points are often similar. Random splitting would cause data leakage.

## How It Works

**How It Works:**

1. Divide the spatial domain into grid blocks
2. Randomly assign entire blocks to train or test
3. All samples within a block stay together

**When to Use:**

- Remote sensing and satellite imagery
- Environmental and ecological modeling
- Real estate price prediction
- Any geospatial machine learning task

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpatialSplitter(Double,Int32,Int32,Int32,Int32,Boolean,Int32)` | Creates a new spatial splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |

