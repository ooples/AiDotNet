---
title: "IFederatedAdapterStrategy<T>"
description: "Interface for federated adapter strategies that enable parameter-efficient fine-tuning (PEFT) in FL."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.Adapters`

Interface for federated adapter strategies that enable parameter-efficient fine-tuning (PEFT) in FL.

## For Beginners

Large AI models (like GPT or LLaMA) have billions of parameters.
Sending all these between clients and server is impractical. Adapters are tiny "add-on" modules
(often <1% of total parameters) that customize the model. Only these adapters are shared
in federated learning, making it practical to fine-tune massive models collaboratively.

## How It Works

Federated adapter strategies allow clients to fine-tune foundation models by training only
small adapter modules (LoRA, prompt tuning) rather than full model parameters. This dramatically
reduces communication costs and enables FL for large language models.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdapterParameterCount` | Gets the total number of trainable adapter parameters. |
| `CompressionRatio` | Gets the compression ratio (adapter params / total model params). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateAdapters(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` | Aggregates adapter parameters from multiple clients. |
| `ExtractAdapterParameters(Vector<>)` | Extracts adapter parameters from a full model parameter vector. |
| `MergeAdapterParameters(Vector<>,Vector<>)` | Merges aggregated adapter parameters back into the full model. |

