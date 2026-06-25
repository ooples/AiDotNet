---
title: "HierarchicalSplitter<T>"
description: "Hierarchical splitter for multi-level nested data structures."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Nested`

Hierarchical splitter for multi-level nested data structures.

## For Beginners

Some data has natural hierarchical structure, like:

- Patients → Hospitals → Regions
- Students → Classes → Schools
- Products → Categories → Departments

## How It Works

This splitter respects the hierarchy, ensuring that splits occur at the
appropriate level to avoid data leakage.

**When to Use:**

- Clinical trials with patients nested in hospitals
- Educational data with students in schools
- Any multi-level grouped data

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HierarchicalSplitter(Double,Int32,Boolean,Int32)` | Creates a new hierarchical splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |
| `WithLevelAssignments(Int32[][])` | Sets the level assignments for samples. |

