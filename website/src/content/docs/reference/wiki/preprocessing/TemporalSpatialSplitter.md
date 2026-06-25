---
title: "TemporalSpatialSplitter<T>"
description: "Combined temporal-spatial splitter for data with both time and location dimensions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.DomainSpecific`

Combined temporal-spatial splitter for data with both time and location dimensions.

## For Beginners

Some data has both time and space components, like:

- Weather observations over time at different locations
- Traffic patterns at intersections over days
- Disease spread tracking

## How It Works

This splitter considers both dimensions to prevent leakage from nearby times AND places.

**Modes:**

- Time-first: Split by time, then spatially within time periods
- Space-first: Split by location, then temporally within regions
- Combined: Use a weighted distance combining both

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TemporalSpatialSplitter(Double,Double,Double,Int32,Int32,Int32,Int32)` | Creates a new temporal-spatial splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |

