---
title: "LeavePGroupsOutSplitter<T>"
description: "Leave-P-Groups-Out cross-validation that uses all combinations of P groups as test sets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.GroupBased`

Leave-P-Groups-Out cross-validation that uses all combinations of P groups as test sets.

## For Beginners

This is similar to Leave-One-Group-Out, but instead of using
one group at a time for testing, it uses all combinations of P groups.

## How It Works

**Example:**
With groups A, B, C, D and p=2:

- Split 1: Test on A,B; Train on C,D
- Split 2: Test on A,C; Train on B,D
- Split 3: Test on A,D; Train on B,C
- Split 4: Test on B,C; Train on A,D
- Split 5: Test on B,D; Train on A,C
- Split 6: Test on C,D; Train on A,B

**When to Use:**

- When you want more test set variety than Leave-One-Group-Out
- When you have enough groups to afford larger test sets
- When evaluating model robustness to different group combinations

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LeavePGroupsOutSplitter(Int32,Int32[],Int32)` | Creates a new Leave-P-Groups-Out splitter. |

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
| `WithGroups(Int32[])` | Sets the group assignments for samples. |

