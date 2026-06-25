---
title: "FineTuningData<T, TInput, TOutput>"
description: "Container for fine-tuning training and evaluation data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Options`

Container for fine-tuning training and evaluation data.

## For Beginners

Think of this as a container that holds all the training
examples the model needs to learn from. Different training methods need different
kinds of examples.

## How It Works

This class holds data for various fine-tuning methods. Different methods use
different subsets of the data:

## Properties

| Property | Summary |
|:-----|:--------|
| `Advantages` | Gets or sets advantages for PPO-style methods. |
| `ChosenOutputs` | Gets or sets the chosen (preferred) outputs for preference learning. |
| `Count` | Gets the number of samples in the dataset. |
| `CritiqueRevisions` | Gets or sets critique-revision pairs for constitutional methods. |
| `DesirabilityLabels` | Gets or sets binary labels indicating if outputs are desirable. |
| `HasDistillationData` | Gets whether this data is suitable for distillation. |
| `HasPairwisePreferenceData` | Gets whether this data is suitable for pairwise preference methods. |
| `HasRLData` | Gets whether this data is suitable for RL methods. |
| `HasRankingData` | Gets whether this data is suitable for ranking methods. |
| `HasSFTData` | Gets whether this data is suitable for SFT. |
| `HasUnpairedPreferenceData` | Gets whether this data is suitable for KTO (unpaired preferences). |
| `Inputs` | Gets or sets the input data samples. |
| `Outputs` | Gets or sets the target outputs for supervised fine-tuning. |
| `RankedOutputs` | Gets or sets ranked outputs for ranking-based methods. |
| `RejectedOutputs` | Gets or sets the rejected outputs for preference learning. |
| `Rewards` | Gets or sets reward values for RL-based methods. |
| `SampleIds` | Gets or sets optional sample identifiers for tracking. |
| `SampleWeights` | Gets or sets optional sample weights for weighted training. |
| `TeacherConfidences` | Gets or sets teacher model confidence scores. |
| `TeacherOutputs` | Gets or sets teacher model logits/outputs for distillation. |
| `Values` | Gets or sets value estimates for critic-based methods. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Double,Nullable<Int32>)` | Splits the data into training and validation sets. |
| `Subset(Int32[])` | Creates a subset of the data for the given indices. |

