---
title: "GraphCodeBERT<T>"
description: "GraphCodeBERT extends CodeBERT by incorporating data flow analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ProgramSynthesis.Engines`

GraphCodeBERT extends CodeBERT by incorporating data flow analysis.

## For Beginners

GraphCodeBERT understands how data flows through code.

While CodeBERT reads code like text, GraphCodeBERT also understands:

- Which variables depend on which others
- How data flows from one function to another
- The relationships and connections in code structure

Think of it like understanding a city:

- CodeBERT sees the streets and buildings (structure)
- GraphCodeBERT also sees how traffic flows and which roads connect (data flow)

This deeper understanding helps with tasks like bug detection and code optimization.

## How It Works

GraphCodeBERT combines source code with data flow information to better understand
code semantics. It uses graph neural networks to model the relationships between
variables, functions, and data dependencies in code.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphCodeBERT` | Initializes a new instance with default architecture settings. |
| `GraphCodeBERT(CodeSynthesisArchitecture<>,ILossFunction<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ITokenizer)` | Initializes a new instance of the `GraphCodeBERT` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `UsesDataFlow` | Gets whether this model uses data flow analysis. |

