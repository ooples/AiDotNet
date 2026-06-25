---
title: "PromptEvalCase"
description: "One labeled example in a prompt-optimization eval set: an input to send the agent and the expected answer to score its response against."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.SelfImproving`

One labeled example in a prompt-optimization eval set: an input to send the agent and the expected answer
to score its response against.

## For Beginners

A practice question with its answer key. The optimizer runs each candidate
prompt against a batch of these to see which prompt gets the most answers right.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PromptEvalCase(String,String)` | Initializes a new eval case. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Expected` | Gets the expected answer. |
| `Input` | Gets the user input to send the agent. |

