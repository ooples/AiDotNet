---
title: "SelfConsistencyStrategy<T>"
description: "Implements Self-Consistency reasoning by sampling multiple reasoning paths and using majority voting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Strategies`

Implements Self-Consistency reasoning by sampling multiple reasoning paths and using majority voting.

## For Beginners

Self-Consistency is like solving a problem multiple times independently
and then comparing answers. If you get the same answer most of the time, you can be confident it's correct.

**How it works:**

1. Generate multiple independent reasoning chains for the same problem
2. Each chain uses slightly different wording/approach (controlled by temperature)
3. Extract the final answer from each chain
4. Use majority voting to pick the most common answer

**Example:**
Problem: "What is 15% of 240?"

Attempt 1: 15% = 0.15, then 0.15 × 240 = 36 ✓
Attempt 2: 240 ÷ 100 × 15 = 36 ✓
Attempt 3: 240 × 15/100 = 36 ✓
Attempt 4: Convert to fraction: 240 × 3/20 = 36 ✓
Attempt 5: (Error) 240 × 1.5 = 360 ✗

Majority vote: "36" appears 4 times, "360" appears 1 time → Answer: "36"

**Why it works:**

- Random errors don't repeat consistently
- Correct reasoning tends to reach the same answer
- Filters out hallucinations and calculation mistakes

**Research basis:**
"Self-Consistency Improves Chain of Thought Reasoning in Language Models" (Wang et al., 2022)
showed significant improvements over standard CoT, especially on reasoning benchmarks.

**When to use:**

- Important decisions where accuracy matters
- Mathematical or logical reasoning
- When you can afford multiple LLM calls
- Problems where errors are common

## How It Works

**Example Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelfConsistencyStrategy(IChatClient<>,IEnumerable<IAgentTool>)` | Initializes a new instance of the `SelfConsistencyStrategy` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `StrategyName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ReasonCoreAsync(String,ReasoningConfig,CancellationToken)` |  |

