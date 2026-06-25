---
title: "ChainOfThoughtBuilder"
description: "Builder for constructing chain-of-thought templates fluently."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.PromptEngineering.Templates`

Builder for constructing chain-of-thought templates fluently.

## Methods

| Method | Summary |
|:-----|:--------|
| `AddExample(ChainOfThoughtExample)` | Adds an example demonstrating step-by-step reasoning. |
| `AddExample(String,String,String)` | Adds an example demonstrating step-by-step reasoning. |
| `Build` | Builds the chain-of-thought template. |
| `WithContext(String)` | Sets additional context for the question. |
| `WithQuestion(String)` | Sets the question to reason about. |

