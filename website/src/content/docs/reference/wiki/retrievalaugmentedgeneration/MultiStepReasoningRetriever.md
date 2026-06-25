---
title: "MultiStepReasoningRetriever<T>"
description: "Multi-step reasoning retriever that breaks down complex queries into sequential steps."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns`

Multi-step reasoning retriever that breaks down complex queries into sequential steps.

## For Beginners

Think of this like solving a mystery by following clues.

Chain-of-Thought (plan everything first):

- Question: "Who invented the transistor and how did it impact computing?"
- Plans: [Find inventor, Find invention date, Find early applications, Find impact]
- Executes all steps

Multi-Step Reasoning (adapt as you learn):

- Question: "Who invented the transistor and how did it impact computing?"
- Step 1: Search "transistor inventor" → Learn about Bell Labs team
- Step 2: Based on Bell Labs finding, search "Bell Labs transistor computing applications"
- Step 3: Based on applications found, search "transistor revolution computer architecture"
- Each step informed by previous discoveries

This is useful when:

- The answer requires building knowledge progressively
- Later steps depend on findings from earlier steps
- You need to adapt the search strategy based on what you find

## How It Works

This advanced retrieval pattern orchestrates multi-step reasoning where each step
builds upon the results of previous steps. Unlike Chain-of-Thought which plans all
steps upfront, multi-step reasoning adapts each step based on what was learned from
previous steps, enabling dynamic problem-solving.

**Example Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiStepReasoningRetriever(IGenerator<>,RetrieverBase<>,Int32)` | Initializes a new instance of the `MultiStepReasoningRetriever` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetermineNextStep(String,String,Int32)` | Determines the next reasoning step based on accumulated knowledge. |
| `ExecuteStep(String,Int32,Dictionary<String,Object>)` | Executes a single reasoning step. |
| `RetrieveMultiStep(String,Int32,Dictionary<String,Object>)` | Retrieves documents using adaptive multi-step reasoning. |
| `SummarizeStepFindings(String,List<Document<>>)` | Summarizes the findings from a reasoning step. |

