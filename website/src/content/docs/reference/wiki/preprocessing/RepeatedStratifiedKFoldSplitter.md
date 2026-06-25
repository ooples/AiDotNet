---
title: "RepeatedStratifiedKFoldSplitter<T>"
description: "Alias for `StratifiedRepeatedKFoldSplitter` with a clearer name ordering."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.CrossValidation`

Alias for `StratifiedRepeatedKFoldSplitter` with a clearer name ordering.

## For Beginners

This is the same as StratifiedRepeatedKFoldSplitter but with
a different name ordering that some users may find more intuitive.

## How It Works

Both names describe the same technique:

- RepeatedStratifiedKFold: "Repeated" (many times) + "Stratified" (class-preserving) + "K-Fold"
- StratifiedRepeatedKFold: "Stratified" (class-preserving) + "Repeated" (many times) + "K-Fold"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RepeatedStratifiedKFoldSplitter(Int32,Int32,Int32)` | Creates a new Repeated Stratified K-Fold splitter. |

