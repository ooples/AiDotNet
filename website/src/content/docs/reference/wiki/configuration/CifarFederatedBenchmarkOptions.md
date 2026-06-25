---
title: "CifarFederatedBenchmarkOptions"
description: "Configuration options for running CIFAR-based federated benchmark suites."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for running CIFAR-based federated benchmark suites.

## For Beginners

This suite tests image classification using the well-known CIFAR datasets.
You point AiDotNet to the extracted CIFAR folder, and it handles the rest.

## How It Works

CIFAR datasets are distributed as binary batch files. AiDotNet loads the dataset from the provided directory,
applies an industry-standard synthetic federated partitioning strategy (for example, Dirichlet label skew),
and evaluates the model with a structured report.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClientCount` | Gets or sets the number of federated clients to simulate (null uses defaults). |
| `DataDirectoryPath` | Gets or sets the dataset directory path containing CIFAR binary files. |
| `DirichletAlpha` | Gets or sets the Dirichlet concentration parameter used when `PartitioningStrategy` is DirichletLabel. |
| `MaxTestSamples` | Gets or sets an optional maximum number of test samples to load (null loads all available). |
| `MaxTrainSamples` | Gets or sets an optional maximum number of training samples to load (null loads all available). |
| `NormalizePixels` | Gets or sets whether pixel values should be normalized to the range [0,1]. |
| `PartitioningStrategy` | Gets or sets the dataset-to-client partitioning strategy. |
| `ShardsPerClient` | Gets or sets how many label shards should be assigned to each client when using `ShardByLabel`. |

