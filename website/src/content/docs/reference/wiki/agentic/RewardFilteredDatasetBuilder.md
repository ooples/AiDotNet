---
title: "RewardFilteredDatasetBuilder"
description: "Builds a `FineTuningDataset` from captured trajectories by keeping only those whose reward meets a threshold and turning each into a (prompt, completion) pair — reward-filtered behavior cloning, the data-preparation half of online LoRA self…"
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Agentic.SelfImproving`

Builds a `FineTuningDataset` from captured trajectories by keeping only those whose reward
meets a threshold and turning each into a (prompt, completion) pair — reward-filtered behavior cloning,
the data-preparation half of online LoRA self-improvement.

## For Beginners

Sift the logbook for the runs that scored well, and from each make a
question→good-answer pair. Those pairs become the lesson plan for fine-tuning the local model on its own
best work.

## How It Works

The prompt is the trajectory's conversation up to (but excluding) its final turn; the completion is the
final answer. The resulting dataset is then handed to the repository's LoRA / fine-tuning trainer to
produce an adapter for the local model — the trainer is the model-layer step; this is the agentic step
that decides *what* to learn from (only high-reward runs).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RewardFilteredDatasetBuilder(Double)` | Initializes a new builder. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Build(IEnumerable<AgentTrajectory>)` | Builds the dataset from the given trajectories, keeping only graded runs at or above the threshold that have both a prompt context and a non-empty completion. |

