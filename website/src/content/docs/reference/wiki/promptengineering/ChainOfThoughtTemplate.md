---
title: "ChainOfThoughtTemplate"
description: "Template that structures prompts for chain-of-thought reasoning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Templates`

Template that structures prompts for chain-of-thought reasoning.

## For Beginners

Helps AI think through problems step by step.

Example:

How it works:

- Structures the prompt to encourage step-by-step thinking
- Includes explicit instructions for showing reasoning
- Helps with math, logic, and complex analysis tasks

## How It Works

This template encourages the model to show its reasoning process step by step,
which often leads to better accuracy on complex tasks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChainOfThoughtTemplate(String)` | Initializes a new instance with a custom template. |
| `ChainOfThoughtTemplate(String,IEnumerable<ChainOfThoughtExample>,String)` | Initializes a new instance with examples for few-shot chain-of-thought. |
| `ChainOfThoughtTemplate(String,String)` | Initializes a new instance of the ChainOfThoughtTemplate class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Builder` | Creates a builder for constructing chain-of-thought prompts. |
| `FormatCore(Dictionary<String,String>)` | Formats the chain-of-thought template. |

