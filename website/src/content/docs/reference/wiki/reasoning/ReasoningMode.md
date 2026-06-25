---
title: "ReasoningMode"
description: "Available reasoning modes that determine how problems are solved."
section: "API Reference"
---

`Enums` · `AiDotNet.Reasoning`

Available reasoning modes that determine how problems are solved.

## For Beginners

Different problems benefit from different thinking approaches.
Just like you might use different study strategies for math vs. essay writing,
the AI can use different reasoning modes for different types of problems.

## Fields

| Field | Summary |
|:-----|:--------|
| `Auto` | Automatically selects the best reasoning mode based on configuration. |
| `ChainOfThought` | Linear step-by-step reasoning. |
| `SelfConsistency` | Solves the problem multiple times and uses majority voting. |
| `TreeOfThoughts` | Explores multiple reasoning paths in a tree structure. |

