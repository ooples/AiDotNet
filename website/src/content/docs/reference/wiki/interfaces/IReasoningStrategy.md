---
title: "IReasoningStrategy<T>"
description: "Defines the contract for reasoning strategies that solve problems through structured thinking."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for reasoning strategies that solve problems through structured thinking.

## For Beginners

A reasoning strategy is like a specific approach or method for solving problems.
Just like you might use different strategies to solve math problems (working backwards, drawing diagrams,
breaking into steps), AI systems can use different reasoning strategies like Chain-of-Thought,
Tree-of-Thoughts, or Self-Consistency.

This interface defines what every reasoning strategy must be able to do:

- Accept a problem or query
- Apply its specific reasoning approach
- Return a structured result with the answer and reasoning trace

Think of it like different cooking methods (baking, frying, steaming) - they're all ways to prepare
food, but each has its own process. Similarly, different reasoning strategies all aim to solve problems,
but each uses a different approach.

## How It Works

**Example Usage:**

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets a description of what this reasoning strategy does and when to use it. |
| `StrategyName` | Gets the name of this reasoning strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ReasonAsync(String,ReasoningConfig,CancellationToken)` | Applies the reasoning strategy to solve a problem or answer a query. |

