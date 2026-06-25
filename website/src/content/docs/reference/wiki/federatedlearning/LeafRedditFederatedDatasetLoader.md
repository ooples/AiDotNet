---
title: "LeafRedditFederatedDatasetLoader"
description: "Loads the LEAF Reddit benchmark JSON files into per-client token-sequence datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Benchmarks.Leaf`

Loads the LEAF Reddit benchmark JSON files into per-client token-sequence datasets.

## For Beginners

Reddit is huge. This loader supports loading a subset of users and sampling per user so
you can run CI-friendly benchmark checks.

## How It Works

The LEAF Reddit preprocessing pipeline stores each sample as a list of token chunks (`x`) and a metadata
object (`y`) containing `target_tokens` (shifted next-token targets) and optional `count_tokens`.
This loader converts each sample into a single fixed-length token sequence paired with a single next-token label
(v1: last non-pad target token).

## Methods

| Method | Summary |
|:-----|:--------|
| `LoadDatasetFromFiles(String,String,LeafFederatedDatasetLoadOptions)` | Loads a LEAF Reddit train dataset and optional test dataset from files. |
| `LoadSplitFromFile(String,LeafFederatedDatasetLoadOptions)` | Loads a LEAF Reddit split (train/test) from a JSON file. |
| `LoadSplitFromJson(String,LeafFederatedDatasetLoadOptions)` | Loads a LEAF Reddit split (train/test) from a JSON string. |

