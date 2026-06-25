---
title: "LeafFederatedBenchmarkOptions"
description: "Configuration options for running LEAF-backed federated benchmark suites."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for running LEAF-backed federated benchmark suites.

## For Beginners

LEAF datasets are stored as JSON files where each "user" corresponds to one federated client.
You provide the file paths, and AiDotNet loads and evaluates the model against the suite.

## How It Works

This options class supplies the dataset context required to run the LEAF suite (train/test JSON split files)
while keeping the user-facing facade surface minimal.

## Properties

| Property | Summary |
|:-----|:--------|
| `LoadOptions` | Gets or sets load options controlling how many users/clients are loaded. |
| `MaxSamplesPerUser` | Gets or sets the maximum number of samples to use per user/client (null uses all available). |
| `TestFilePath` | Gets or sets the optional path to the LEAF test split JSON file. |
| `TrainFilePath` | Gets or sets the path to the LEAF train split JSON file. |

