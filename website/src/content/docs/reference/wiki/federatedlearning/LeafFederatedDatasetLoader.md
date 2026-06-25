---
title: "LeafFederatedDatasetLoader<T>"
description: "Loads LEAF benchmark JSON files into per-client datasets suitable for federated learning simulation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Benchmarks.Leaf`

Loads LEAF benchmark JSON files into per-client datasets suitable for federated learning simulation.

## For Beginners

This loader reads one LEAF JSON file and converts it into a set of
"client datasets" — one dataset per user — so federated learning trainers can run simulations.

## How It Works

LEAF stores federated datasets in a JSON structure containing:

- `users`: user/client IDs
- `num_samples`: declared sample counts per user
- `user_data`: per-user objects with `x` (features) and `y` (labels)

## Methods

| Method | Summary |
|:-----|:--------|
| `LoadDatasetFromFiles(String,String,LeafFederatedDatasetLoadOptions)` | Loads a LEAF train dataset and optional test dataset from files. |
| `LoadSplitFromFile(String,LeafFederatedDatasetLoadOptions)` | Loads a LEAF split (train/test) from a JSON file. |
| `LoadSplitFromJson(String,LeafFederatedDatasetLoadOptions)` | Loads a LEAF split (train/test) from a JSON string. |

