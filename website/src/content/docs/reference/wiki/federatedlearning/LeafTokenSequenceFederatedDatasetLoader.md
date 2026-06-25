---
title: "LeafTokenSequenceFederatedDatasetLoader"
description: "Loads LEAF-style JSON files that store token sequences (`x`) and next-token labels (`y`)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Benchmarks.Leaf`

Loads LEAF-style JSON files that store token sequences (`x`) and next-token labels (`y`).

## For Beginners

Some federated text benchmarks are easiest to represent as "predict the next token".
This loader keeps the per-user splits intact so federated simulations match the benchmark partitioning.

## How It Works

This loader is intentionally generic: it expects the standard LEAF container shape
(`users`/`num_samples`/`user_data`) where each user's `x` is a list of token sequences
(arrays of strings) and each `y` is a list of label tokens (strings).

## Methods

| Method | Summary |
|:-----|:--------|
| `LoadDatasetFromFiles(String,String,LeafFederatedDatasetLoadOptions)` | Loads a LEAF token-sequence train dataset and optional test dataset from files. |
| `LoadSplitFromFile(String,LeafFederatedDatasetLoadOptions)` | Loads a LEAF token-sequence split (train/test) from a JSON file. |
| `LoadSplitFromJson(String,LeafFederatedDatasetLoadOptions)` | Loads a LEAF token-sequence split (train/test) from a JSON string. |

