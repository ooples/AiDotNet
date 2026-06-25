---
title: "LogicalReasoner<T>"
description: "LogicalReasoner<T> — Models & Types in AiDotNet.Reasoning.DomainSpecific."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.DomainSpecific`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LogicalReasoner(IChatClient<>,Boolean)` | Initializes a new instance of the `LogicalReasoner` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnalyzeRelationshipAsync(String,String,ReasoningConfig,CancellationToken)` | Determines the logical relationship between statements. |
| `DetectFallaciesAsync(String,ReasoningConfig,CancellationToken)` | Identifies logical fallacies in reasoning. |
| `EvaluateArgumentAsync(String,List<String>,String,ReasoningConfig,CancellationToken)` | Evaluates the validity of a logical argument. |
| `ProveAsync(String,List<String>,ReasoningConfig,CancellationToken)` | Constructs a formal proof for a logical statement. |
| `SolveAsync(String,String,ReasoningConfig,Boolean,CancellationToken)` | Solves a logical reasoning problem. |
| `SolvePuzzleAsync(String,ReasoningConfig,CancellationToken)` | Solves a logic puzzle with constraints. |

