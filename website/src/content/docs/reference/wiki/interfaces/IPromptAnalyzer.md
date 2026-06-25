---
title: "IPromptAnalyzer"
description: "Defines the contract for analyzing prompts before sending them to language models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for analyzing prompts before sending them to language models.

## For Beginners

A prompt analyzer is like a spell-checker and cost calculator for your prompts.

Before sending a prompt to an AI model (which costs money), the analyzer tells you:

- How many tokens it uses (tokens = cost)
- Estimated API cost in dollars
- How complex the prompt is
- Any potential problems (missing variables, too long, etc.)

Example workflow:

Benefits:

- Cost control: Know costs before making API calls
- Optimization: Find prompts that are too long or complex
- Debugging: Catch issues before they cause errors
- Budgeting: Track and forecast API spending

## How It Works

A prompt analyzer computes metrics and validates prompts to help developers understand
and optimize their prompts before incurring API costs. Analysis includes token counting,
cost estimation, complexity measurement, and issue detection.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this analyzer implementation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Analyze(String)` | Analyzes a prompt and returns detailed metrics. |
| `AnalyzeAsync(String,CancellationToken)` | Analyzes a prompt asynchronously for use in async workflows. |
| `ValidatePrompt(String,ValidationOptions)` | Validates a prompt for potential issues. |

