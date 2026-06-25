---
title: "Sent140FederatedBenchmarkOptions"
description: "Configuration options for running the Sent140 LEAF federated benchmark suite."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for running the Sent140 LEAF federated benchmark suite.

## For Beginners

You provide the train/test JSON files, and AiDotNet loads the per-user partitions,
tokenizes the tweet text, and evaluates your model.

## How It Works

Sent140 is a federated sentiment classification benchmark derived from tweets. LEAF stores the dataset as JSON
where each "user" corresponds to a federated client.

## Properties

| Property | Summary |
|:-----|:--------|
| `LoadOptions` | Gets or sets load options controlling how many users/clients are loaded. |
| `MaxSamplesPerUser` | Gets or sets the maximum number of samples to use per user/client (null uses all available). |
| `MaxSequenceLength` | Gets or sets the maximum token sequence length for each tweet after tokenization. |
| `TestFilePath` | Gets or sets the optional path to the Sent140 test split JSON file. |
| `TokenizerTrainingSampleCount` | Gets or sets the maximum number of texts used to train a default WordPiece tokenizer when needed. |
| `TokenizerVocabularySize` | Gets or sets the desired WordPiece vocabulary size when the model result does not already provide a tokenizer. |
| `TrainFilePath` | Gets or sets the path to the Sent140 train split JSON file. |

