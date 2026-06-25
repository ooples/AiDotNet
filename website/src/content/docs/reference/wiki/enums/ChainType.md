---
title: "ChainType"
description: "Represents different types of chains for composing language model operations."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Represents different types of chains for composing language model operations.

## For Beginners

Chains connect multiple language model operations together, like building blocks.

Think of chains like assembly lines:

- Each step does a specific task
- Output from one step becomes input for the next
- Complex workflows are built from simple components
- The final result is the combination of all steps

For example, a research assistant might:

1. Search for relevant documents (Step 1)
2. Summarize each document (Step 2)
3. Combine summaries into a final report (Step 3)

Chains make this easy to build, test, and modify.

## Fields

| Field | Summary |
|:-----|:--------|
| `Conditional` | Conditional chain with branching logic based on intermediate results. |
| `Loop` | Looping chain that repeats operations until a condition is met. |
| `MapReduce` | Map-reduce chain that processes collections by mapping operations and reducing results. |
| `Parallel` | Parallel chain that executes multiple independent operations simultaneously. |
| `Router` | Router chain that directs inputs to specialized chains based on content. |
| `Sequential` | Simple sequential chain where output of each step feeds into the next. |

