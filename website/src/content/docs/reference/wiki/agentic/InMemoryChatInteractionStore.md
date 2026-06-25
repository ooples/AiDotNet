---
title: "InMemoryChatInteractionStore"
description: "A process-local `IChatInteractionStore` holding recorded chat interactions in memory."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Pipeline`

A process-local `IChatInteractionStore` holding recorded chat interactions in memory. Ideal
for tests and within-process record/replay; contents are lost when the process exits.

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Save(String,ChatResponse)` |  |
| `TryGet(String,ChatResponse)` |  |

