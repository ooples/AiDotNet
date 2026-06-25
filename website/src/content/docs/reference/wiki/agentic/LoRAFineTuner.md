---
title: "LoRAFineTuner"
description: "Runs the online self-improvement loop's final step: fine-tune (e.g."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Agentic.SelfImproving`

Runs the online self-improvement loop's final step: fine-tune (e.g. LoRA-adapt) a model on a reward-filtered
`FineTuningDataset` by converting it to supervised data and invoking a fine-tuner. This closes
the loop capture → evaluate → reward-filter → *train*.

## For Beginners

Hand it your "good runs" dataset, a fine-tuner, and a model, and it trains the
model to do more of what worked.

## How It Works

The fine-tuner is any `FineTuningBase` over string in/out (notably
`SupervisedFineTuning`) and the model any `IFullModel` that consumes
text. The actual training (LoRA via the configured `FineTuningOptions`) and its hardware cost
live in the fine-tuner; this is the thin, deterministic bridge from the agentic dataset to that call.

## Methods

| Method | Summary |
|:-----|:--------|
| `FineTuneFromDatasetAsync(FineTuningBase<,String,String>,IFullModel<,String,String>,FineTuningDataset,CancellationToken)` | Fine-tunes a model on a reward-filtered dataset. |
| `TrainTensorModelOnDataset(NeuralNetworkBase<>,IGenerationTokenizer,Int32,Int32,FineTuningDataset,Int32,CancellationToken)` | Trains a *tensor* model end-to-end on a reward-filtered dataset: the dataset text is tokenized into next-token tensor supervision (via `TextTensorDatasetConverter`) and the model is trained on it for the given number of epochs using the net… |

