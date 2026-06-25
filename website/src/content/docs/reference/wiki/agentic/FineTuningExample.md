---
title: "FineTuningExample"
description: "One supervised fine-tuning example distilled from a high-reward trajectory: the prompt the agent saw and the completion it produced that earned a good score."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.SelfImproving`

One supervised fine-tuning example distilled from a high-reward trajectory: the prompt the agent saw and
the completion it produced that earned a good score.

## For Beginners

A "do it like this" training pair — the question plus a known-good answer —
harvested from a run that scored well. Many of these teach a local model to imitate its own best behavior.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FineTuningExample(String,String,Double)` | Initializes a new example. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Completion` | Gets the completion to learn. |
| `Prompt` | Gets the input/context the agent was given. |
| `Reward` | Gets the reward the originating trajectory earned. |

