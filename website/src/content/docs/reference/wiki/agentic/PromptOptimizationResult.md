---
title: "PromptOptimizationResult"
description: "The outcome of prompt optimization: the best-scoring prompt plus the full ranked list of candidates."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.SelfImproving`

The outcome of prompt optimization: the best-scoring prompt plus the full ranked list of candidates.

## For Beginners

After trying each prompt on the practice questions, this tells you which prompt
scored best (use that one) and shows every prompt's score so you can see the spread.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PromptOptimizationResult(String,Double,IReadOnlyList<ScoredPrompt>)` | Initializes a new result. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BestPrompt` | Gets the highest-scoring prompt. |
| `BestScore` | Gets the best prompt's mean score. |
| `Candidates` | Gets all candidates with their scores, ranked best first. |

