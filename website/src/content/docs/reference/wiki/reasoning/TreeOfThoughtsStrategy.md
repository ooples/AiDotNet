---
title: "TreeOfThoughtsStrategy<T>"
description: "Implements Tree-of-Thoughts (ToT) reasoning that explores multiple reasoning paths in a tree structure."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Strategies`

Implements Tree-of-Thoughts (ToT) reasoning that explores multiple reasoning paths in a tree structure.

## For Beginners

Tree-of-Thoughts (ToT) is like exploring a maze where you can try
multiple paths and backtrack if you hit a dead end. Unlike Chain-of-Thought which follows one
linear path, ToT explores a tree of possibilities.

**How it works:**
```
Problem: "How can we reduce carbon emissions?"

├─ Renewable Energy (score: 0.9)
│ ├─ Solar panels on buildings (score: 0.85)
│ ├─ Wind farm expansion (score: 0.80)
│ └─ Hydroelectric upgrades (score: 0.75)
│
├─ Transportation (score: 0.85)
│ ├─ Electric vehicles (score: 0.90) ← Best path
│ └─ Public transit (score: 0.82)
│
└─ Industrial (score: 0.75)
├─ Carbon capture (score: 0.70)
└─ Process efficiency (score: 0.65)
```

The search algorithm explores this tree to find the best reasoning path.

**Key features:**

- **Explores multiple paths**: Not limited to one direction
- **Can backtrack**: If a path looks bad, try another
- **Evaluation at each step**: Score thoughts as you go
- **Configurable search**: BFS, Beam Search, or other algorithms

**Compared to other strategies:**

- **Chain-of-Thought**: Linear, one path only
- **Self-Consistency**: Multiple independent paths, no tree structure
- **Tree-of-Thoughts**: Structured exploration with evaluation and backtracking

**Research basis:**
"Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2023)
showed significant improvements on planning, math, and creative tasks.

**When to use:**

- Complex problems with multiple viable approaches
- When you need to explore and compare alternatives
- Planning tasks with branching possibilities
- Creative problem-solving
- Strategic decision-making

## How It Works

**Example Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TreeOfThoughtsStrategy(IChatClient<>,IEnumerable<IAgentTool>,SearchAlgorithmType)` | Initializes a new instance of the `TreeOfThoughtsStrategy` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `StrategyName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CountNodesInTree(ThoughtNode<>)` | Counts total nodes in the tree (DFS). |
| `CreateSearchAlgorithm(SearchAlgorithmType)` | Creates the appropriate search algorithm based on type. |
| `ExtractFinalAnswerAsync(List<ThoughtNode<>>,String,ReasoningConfig,CancellationToken)` | Extracts or generates the final answer from the best reasoning path. |
| `ReasonCoreAsync(String,ReasoningConfig,CancellationToken)` |  |

