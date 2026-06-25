---
title: "HoldoutSplitter<T>"
description: "Creates multiple independent holdout test sets for robust evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Basic`

Creates multiple independent holdout test sets for robust evaluation.

## For Beginners

This splitter creates multiple independent train/test splits,
where each test set (holdout) is completely separate. Unlike K-Fold where test sets
don't overlap, holdout splits can have different random samples each time.

## How It Works

**When to Use:**

- When you need multiple independent evaluations
- For statistical significance testing
- When you want to assess model stability across different splits

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HoldoutSplitter(Int32,Double,Boolean,Int32)` | Creates a new holdout splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `NumSplits` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSplits(Matrix<>,Vector<>)` |  |
| `Split(Matrix<>,Vector<>)` |  |

