---
title: "FineTuningDataConverter"
description: "Bridges the self-improving layer to the fine-tuning framework: converts a reward-filtered `FineTuningDataset` (prompt → good-completion pairs) into the framework's supervised `FineTuningData` shape so it can be handed to `SupervisedFineTuni…"
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Agentic.SelfImproving`

Bridges the self-improving layer to the fine-tuning framework: converts a reward-filtered
`FineTuningDataset` (prompt → good-completion pairs) into the framework's supervised
`FineTuningData` shape so it can be handed to
`SupervisedFineTuning` (or any SFT-capable fine-tuner) to fine-tune / LoRA-adapt a model on the
agent's own best behavior.

## For Beginners

Turns the "good runs" you collected into the exact format the fine-tuning
trainer expects, so you can teach a model to imitate its own successes.

## How It Works

Prompts become `Inputs`, completions become `Outputs`, and each example's reward is carried as a
`SampleWeight` so the trainer can weight higher-reward examples more heavily. The resulting data uses
`string` inputs/outputs; a model that consumes text (or a tokenization step in front of a tensor
model) completes the loop — the training execution itself is a model-pipeline concern.

## Methods

| Method | Summary |
|:-----|:--------|
| `ToSupervisedData(FineTuningDataset)` | Converts a reward-filtered dataset to supervised fine-tuning data (string in/out), carrying rewards as sample weights. |

