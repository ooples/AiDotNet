---
title: "FineTuningDataset"
description: "A set of reward-filtered `FineTuningExample`s ready to hand to a LoRA / fine-tuning trainer — the bridge from \"the agent did well on these runs\" to \"make the local model better at them\"."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.SelfImproving`

A set of reward-filtered `FineTuningExample`s ready to hand to a LoRA / fine-tuning trainer —
the bridge from "the agent did well on these runs" to "make the local model better at them".

## For Beginners

A study packet of known-good question/answer pairs. Feed it to the existing
LoRA fine-tuning trainer and the local model improves at the kinds of task it already handled well —
learning from its own successes (reward-filtered behavior cloning).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FineTuningDataset(IReadOnlyList<FineTuningExample>)` | Initializes a new dataset. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the number of examples. |
| `Examples` | Gets the fine-tuning examples. |
| `MeanReward` | Gets the mean reward across the examples (0 when empty). |

