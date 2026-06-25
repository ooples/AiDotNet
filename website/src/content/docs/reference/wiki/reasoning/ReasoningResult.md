---
title: "ReasoningResult<T>"
description: "Represents the complete result of a reasoning process, including the answer, reasoning chain, and performance metrics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Models`

Represents the complete result of a reasoning process, including the answer, reasoning chain, and performance metrics.

## For Beginners

Think of ReasoningResult as the complete package you get back after the AI
solves a problem. It's like when you finish a homework problem and you have:

- The final answer
- All your work showing how you got there
- Notes about which parts you checked or corrected
- How long it took you
- How confident you are about the answer

This class bundles all of that information together in one place, making it easy to work with
the results of reasoning.

## How It Works

**Example Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReasoningResult` | Initializes a new instance of the `ReasoningResult` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlternativeChains` | Alternative reasoning chains that were explored (for strategies like Tree-of-Thoughts). |
| `ConfidenceScores` | Vector of confidence scores across all attempted reasoning paths. |
| `ErrorMessage` | Error message if the reasoning failed (null if successful). |
| `FinalAnswer` | The final answer or solution from the reasoning process. |
| `Metadata` | Additional metadata or context about this reasoning result. |
| `Metrics` | Performance metrics and statistics about the reasoning process. |
| `OverallConfidence` | Overall confidence score for the final answer (0.0 to 1.0). |
| `ReasoningChain` | The complete chain of reasoning steps that led to the final answer. |
| `StrategyUsed` | The strategy that was used for reasoning (e.g., "Chain-of-Thought", "Tree-of-Thoughts"). |
| `Success` | Whether the reasoning process completed successfully. |
| `ToolsUsed` | Tools or external resources that were used during reasoning. |
| `TotalDuration` | Total time spent on the reasoning process. |
| `VerificationFeedback` | Verification results and feedback from critic models. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSummary` | Creates a summary string of the reasoning result. |
| `ToString` | Returns a string representation of this reasoning result. |

