---
title: "DiscreteSearchOptimizer<T>"
description: "Optimizer that uses discrete search to find better prompts by testing variations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Optimization`

Optimizer that uses discrete search to find better prompts by testing variations.

## For Beginners

Tries different versions of prompt parts and picks what works best.

Example:
```cs
var optimizer = new DiscreteSearchOptimizer<double>();

// Evaluation function that returns accuracy
double Evaluate(string prompt)
{
var correct = 0;
foreach (var test in testCases)
{
var result = model.Generate(prompt + test.Input);
if (result == test.Expected) correct++;
}
return correct / (double)testCases.Count;
}

var optimized = optimizer.Optimize(
initialPrompt: "Classify the sentiment:",
evaluationFunction: Evaluate,
maxIterations: 50
);

// Returns best-performing prompt variation
```

## How It Works

This optimizer generates variations of prompt components and tests combinations to find
the best-performing prompt. It's interpretable and systematic.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiscreteSearchOptimizer` | Initializes a new instance of the DiscreteSearchOptimizer class with default variations. |
| `DiscreteSearchOptimizer(IEnumerable<String>,IEnumerable<String>)` | Initializes a new instance of the DiscreteSearchOptimizer class with custom variations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddFormatVariations(String[])` | Adds custom format variations to try during optimization. |
| `AddInstructionVariations(String[])` | Adds custom instruction variations to try during optimization. |
| `OptimizeCore(String,Func<String,>,Int32)` | Optimizes the prompt using discrete search. |
| `OptimizeCoreAsync(String,Func<String,Task<>>,Int32,CancellationToken)` | Optimizes the prompt asynchronously using discrete search. |

