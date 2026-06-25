---
title: "StackOverflowFederatedBenchmarkOptions"
description: "Configuration options for running the StackOverflow federated benchmark suite."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for running the StackOverflow federated benchmark suite.

## For Beginners

You provide the train/test JSON files, and AiDotNet loads the per-user partitions,
builds a vocabulary (with safe defaults), and evaluates your model on a standardized next-token task.

## How It Works

StackOverflow is a large-scale federated text benchmark commonly used for next-token prediction.
This v1 suite expects a LEAF-style JSON container with per-user token sequences.

## Properties

| Property | Summary |
|:-----|:--------|
| `LoadOptions` | Gets or sets load options controlling how many users/clients are loaded. |
| `MaxSamplesPerUser` | Gets or sets the maximum number of samples to use per user/client (null uses all available). |
| `MaxVocabularySize` | Gets or sets the maximum vocabulary size used for token-to-ID mapping. |
| `SequenceLength` | Gets or sets the fixed token sequence length used as model input. |
| `TestFilePath` | Gets or sets the optional path to the StackOverflow test split JSON file. |
| `TrainFilePath` | Gets or sets the path to the StackOverflow train split JSON file. |
| `VocabularyTrainingSampleCount` | Gets or sets the maximum number of sequences used to build the default vocabulary. |

