---
title: "ToolAugmentedReasoningRetriever<T>"
description: "Tool-augmented reasoning retriever that can use external tools during reasoning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns`

Tool-augmented reasoning retriever that can use external tools during reasoning.

## For Beginners

Think of this like a researcher with access to specialized equipment.

Without tools:

- "What is the compound annual growth rate of solar installations from 2015 to 2023?"
- Can only retrieve documents about growth rates

With tools:

- Retrieves data: 2015: 50 GW, 2023: 400 GW
- Recognizes calculation needed
- Uses calculator tool: CAGR = (400/50)^(1/8) - 1 = 29.4%
- Incorporates calculation into answer

Supported tool types:

- Calculator: Mathematical computations
- Code: Execute code for data processing
- Custom: User-defined tools

## How It Works

This pattern extends multi-step reasoning by incorporating external tools such as
calculators, code interpreters, or specialized APIs. The system can recognize when
a tool is needed, invoke it, and incorporate the results into the reasoning process.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ToolAugmentedReasoningRetriever(IGenerator<>,RetrieverBase<>)` | Initializes a new instance of the `ToolAugmentedReasoningRetriever` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnalyzeToolNeeds(String,List<Document<>>)` | Analyzes whether tools are needed for the query. |
| `EvaluateSimpleMathExpression(String)` | Simple math expression evaluator (basic implementation). |
| `RegisterDefaultTools` | Registers default tools (calculator, string operations). |
| `RegisterTool(String,Func<String,String>)` | Registers a custom tool for use during reasoning. |
| `RetrieveWithTools(String,Int32,Dictionary<String,Object>)` | Retrieves documents using tool-augmented reasoning. |

