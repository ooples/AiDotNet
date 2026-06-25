---
title: "RedditFederatedBenchmarkOptions"
description: "Configuration options for running the Reddit federated benchmark suite."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for running the Reddit federated benchmark suite.

## For Beginners

You provide the train/test JSON files, and AiDotNet loads the per-user partitions,
builds a vocabulary (with safe defaults), and evaluates your model on a standardized next-token task.

## How It Works

Reddit is a large-scale federated text benchmark. This suite uses a token-sequence formulation (next-token
prediction) and evaluates models without exposing model internals.

## Properties

| Property | Summary |
|:-----|:--------|
| `LoadOptions` | Gets or sets load options controlling how many users/clients are loaded. |
| `MaxSamplesPerUser` | Gets or sets the maximum number of samples to use per user/client (null uses all available). |
| `MaxVocabularySize` | Gets or sets the maximum vocabulary size used for token-to-ID mapping. |
| `SequenceLength` | Gets or sets the fixed token sequence length used as model input. |
| `TestFilePath` | Gets or sets the optional path to the Reddit test split JSON file. |
| `TrainFilePath` | Gets or sets the path to the Reddit train split JSON file. |
| `VocabularyTrainingSampleCount` | Gets or sets the maximum number of sequences used to build the default vocabulary. |

