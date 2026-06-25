---
title: "MemoryAugmentationOptions"
description: "Settings for `MemoryAugmentedAgent`: how many memories to recall, the minimum relevance to include, and the heading used when injecting them into the conversation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Agentic.Memory`

Settings for `MemoryAugmentedAgent`: how many memories to recall, the minimum relevance to
include, and the heading used when injecting them into the conversation.

## For Beginners

These control how the assistant uses its long-term notes before answering:
how many of the most relevant notes to pull in (`TopK`), how relevant a note must be to bother
including it (`MinScore`), and the little title that introduces them.

## Properties

| Property | Summary |
|:-----|:--------|
| `Header` | Gets or sets the heading prepended to the recalled memories. |
| `MinScore` | Gets or sets the minimum relevance score a memory must reach to be injected. |
| `TopK` | Gets or sets the maximum number of memories to recall and inject. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultHeader` | The default heading prepended to recalled memories. |
| `DefaultTopK` | The default number of memories to recall per turn. |

