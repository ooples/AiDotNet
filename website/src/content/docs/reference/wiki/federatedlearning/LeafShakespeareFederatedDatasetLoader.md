---
title: "LeafShakespeareFederatedDatasetLoader"
description: "Loads the LEAF Shakespeare benchmark JSON files into per-client datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Benchmarks.Leaf`

Loads the LEAF Shakespeare benchmark JSON files into per-client datasets.

## For Beginners

This loader reads LEAF JSON and returns one dataset per user so federated learning
simulations match the benchmark's per-user partitioning.

## How It Works

Shakespeare is a federated next-character prediction benchmark. LEAF stores each sample as a fixed-length
character window (`x`) and the next character (`y`).

## Methods

| Method | Summary |
|:-----|:--------|
| `LoadDatasetFromFiles(String,String,LeafFederatedDatasetLoadOptions)` | Loads a LEAF Shakespeare train dataset and optional test dataset from files. |
| `LoadSplitFromFile(String,LeafFederatedDatasetLoadOptions)` | Loads a LEAF Shakespeare split (train/test) from a JSON file. |
| `LoadSplitFromJson(String,LeafFederatedDatasetLoadOptions)` | Loads a LEAF Shakespeare split (train/test) from a JSON string. |

