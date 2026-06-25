---
title: "ShakespeareFederatedBenchmarkOptions"
description: "Configuration options for running the Shakespeare LEAF federated benchmark suite."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for running the Shakespeare LEAF federated benchmark suite.

## For Beginners

You provide the train/test JSON files, and AiDotNet loads the per-user partitions,
tokenizes the character sequences, and evaluates your model.

## How It Works

Shakespeare is a federated next-character prediction benchmark. LEAF stores the dataset as JSON where each
"user" corresponds to a federated client and each sample is a fixed-length text window with the next character
as the label.

## Properties

| Property | Summary |
|:-----|:--------|
| `LoadOptions` | Gets or sets load options controlling how many users/clients are loaded. |
| `MaxSamplesPerUser` | Gets or sets the maximum number of samples to use per user/client (null uses all available). |
| `SequenceLength` | Gets or sets the fixed character window length used as model input. |
| `TestFilePath` | Gets or sets the optional path to the Shakespeare test split JSON file. |
| `TrainFilePath` | Gets or sets the path to the Shakespeare train split JSON file. |

