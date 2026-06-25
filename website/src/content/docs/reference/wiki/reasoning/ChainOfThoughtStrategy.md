---
title: "ChainOfThoughtStrategy<T>"
description: "Implements Chain-of-Thought (CoT) reasoning that solves problems through explicit step-by-step thinking."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Strategies`

Implements Chain-of-Thought (CoT) reasoning that solves problems through explicit step-by-step thinking.

## For Beginners

Chain-of-Thought (CoT) is a reasoning approach where the AI explicitly
shows its work, step by step, similar to how you would solve a math problem by writing down each step.

**How it works:**
Given: "What is 15% of 240?"

Step 1: Convert percentage to decimal

- 15% = 15/100 = 0.15

Step 2: Multiply by the number

- 0.15 × 240 = 36

Step 3: State the answer

- The answer is 36

**Why it's effective:**

- Makes reasoning transparent and verifiable
- Catches logical errors early
- Improves accuracy on complex problems
- Allows debugging when answers are wrong

**Based on research:**
"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)
showed 3-5x improvements on reasoning tasks when models show their work.

## How It Works

**Example Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChainOfThoughtStrategy(IChatClient<>,IEnumerable<IAgentTool>,Boolean)` | Initializes a new instance of the `ChainOfThoughtStrategy` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `StrategyName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildChainOfThoughtPrompt(String,ReasoningConfig)` | Builds the prompt that instructs the LLM to use Chain-of-Thought reasoning. |
| `CreateReasoningStep(Int32,String)` | Creates a reasoning step with default high confidence. |
| `ExtractFinalAnswer(String,List<ReasoningStep<>>)` | Extracts the final answer from the response. |
| `ExtractJsonFromResponse(String)` | Extracts JSON content from markdown code blocks. |
| `ParseReasoningSteps(String,Int32)` | Parses reasoning steps from the LLM response. |
| `ParseWithRegex(String,Int32)` | Fallback regex parser for non-JSON responses. |
| `ReasonCoreAsync(String,ReasoningConfig,CancellationToken)` |  |

