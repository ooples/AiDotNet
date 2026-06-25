---
title: "CodeReasoner<T>"
description: "CodeReasoner<T> — Models & Types in AiDotNet.Reasoning.DomainSpecific."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.DomainSpecific`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CodeReasoner(IChatClient<>,IEnumerable<IAgentTool>)` | Initializes a new instance of the `CodeReasoner` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DebugCodeAsync(String,String,ReasoningConfig,CancellationToken)` | Debugs code by analyzing errors and suggesting fixes. |
| `DetectProgrammingLanguage(String)` | Detects the programming language from text. |
| `ExplainCodeAsync(String,ReasoningConfig,CancellationToken)` | Explains how existing code works. |
| `ExtractCode(String)` | Extracts code blocks from markdown-formatted text. |
| `GenerateCodeAsync(String,String,ReasoningConfig,CancellationToken)` | Generates code with step-by-step explanation. |
| `LooksLikeCode(String)` | Heuristic check if text looks like code. |
| `SolveAsync(String,ReasoningConfig,Boolean,CancellationToken)` | Solves a code-related problem using reasoning. |

