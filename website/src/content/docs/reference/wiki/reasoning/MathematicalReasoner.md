---
title: "MathematicalReasoner<T>"
description: "Specialized reasoner for mathematical problems using verified reasoning and external verification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.DomainSpecific`

Specialized reasoner for mathematical problems using verified reasoning and external verification.

## For Beginners

MathematicalReasoner is like a math tutor that not only solves problems
but also checks its work with a calculator and verifies each step makes sense.

**What makes it special for math:**

- Uses Chain-of-Thought to show work step-by-step
- Verifies calculations with CalculatorVerifier
- Can use Self-Consistency for important problems
- Critic model checks mathematical reasoning
- Self-refinement fixes calculation errors

**Example workflow:**
Problem: "If John has 15 apples and gives away 40%, how many does he have left?"

1. Generate reasoning steps
2. Step 1: "Calculate 40% of 15 = 6"

→ CalculatorVerifier: ✓ Correct

3. Step 2: "Subtract: 15 - 6 = 9"

→ CalculatorVerifier: ✓ Correct

4. Step 3: "John has 9 apples left"
5. All steps verified → High confidence answer

**Used for benchmarks:**

- GSM8K (grade school math)
- MATH (competition mathematics)
- Any mathematical reasoning tasks

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MathematicalReasoner(IChatClient<>,IEnumerable<IAgentTool>)` | Initializes a new instance of the `MathematicalReasoner` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractNumericalAnswer(ReasoningResult<>)` | Extracts numerical answer from reasoning result for benchmark evaluation. |
| `RefineChainWithCalculationFeedbackAsync(ReasoningChain<>,ReasoningConfig,CancellationToken)` | Refines a chain based on calculation verification feedback. |
| `SolveAsync(String,ReasoningConfig,Boolean,Boolean,CancellationToken)` | Solves a mathematical problem using verified reasoning. |
| `VerifyAndRefineAsync(ReasoningResult<>,ReasoningConfig,CancellationToken)` | Verifies and refines the reasoning result. |

