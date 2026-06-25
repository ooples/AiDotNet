---
title: "ReasoningChain<T>"
description: "Represents a complete chain of reasoning steps from problem to solution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Models`

Represents a complete chain of reasoning steps from problem to solution.

## For Beginners

A reasoning chain is like showing your complete work on a problem,
from start to finish. Just like in math class where you write out all your steps:

Problem: "What is 15% of 240?"
Step 1: Convert percentage to decimal
Step 2: Multiply
Step 3: State the answer

The ReasoningChain class keeps track of all these steps together, along with scores that
tell you how confident the AI is about each step. It uses a Vector to store scores efficiently,
which is important for machine learning operations.

## How It Works

**Example Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReasoningChain` | Initializes a new instance of the `ReasoningChain` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CompletedAt` | When this reasoning chain was completed. |
| `Duration` | Total time spent on this reasoning chain. |
| `FinalAnswer` | The final answer or conclusion from this reasoning chain. |
| `IsFullyVerified` | Whether every step in this chain has been verified. |
| `Metadata` | Additional metadata or context for this reasoning chain. |
| `OverallScore` | Overall confidence score for the entire reasoning chain. |
| `Query` | The original query or problem that this reasoning chain addresses. |
| `StartedAt` | When this reasoning chain was started. |
| `StepScores` | Vector of confidence scores for each step, enabling efficient ML operations. |
| `Steps` | The ordered list of reasoning steps in this chain. |
| `TotalRefinements` | Total number of refinements made across all steps in this chain. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddStep(ReasoningStep<>)` | Adds a reasoning step to this chain. |
| `GetAverageScore` | Gets the average confidence score across all steps. |
| `GetMinimumScore` | Gets the minimum confidence score across all steps. |
| `ToString` | Returns a formatted string representation of the entire reasoning chain. |

