---
title: "ThoughtNode<T>"
description: "Represents a node in a tree of thoughts, used for exploring multiple reasoning paths."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Reasoning.Models`

Represents a node in a tree of thoughts, used for exploring multiple reasoning paths.

## For Beginners

Imagine you're solving a complex problem and at each step, you could go
in several different directions. A ThoughtNode represents one possible "thought" or direction
you might explore.

Think of it like a choose-your-own-adventure book:

- Each page (node) presents a situation (the thought)
- Each page might have several choices that lead to different pages (children)
- You can trace back through your choices (parent links)
- Some paths lead to good endings (high scores), others to bad ones (low scores)

Tree-of-Thoughts reasoning builds a tree of these nodes, exploring different paths
and choosing the best ones. It's more sophisticated than just following one path
(Chain-of-Thought) because it can compare alternatives and backtrack if needed.

## How It Works

**Example Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ThoughtNode` | Initializes a new instance of the `ThoughtNode` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Children` | Child nodes representing alternative next steps from this thought. |
| `Depth` | Depth level of this node in the tree (root = 0). |
| `EvaluationScore` | Quality score evaluating how promising this thought is (typically 0.0 to 1.0). |
| `IsTerminal` | Whether this node represents a complete solution or terminal state. |
| `IsVisited` | Whether this node has been visited during tree exploration. |
| `Metadata` | Additional context or metadata specific to this thought. |
| `Parent` | The parent node that led to this thought (null for root node). |
| `PathLength` | Gets the number of nodes in the complete path from root to this node. |
| `PathScores` | The complete path from root to this node as a Vector of scores. |
| `Thought` | The thought or reasoning content at this node. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CheckIsTerminalByHeuristic` | Checks if this node appears to be terminal based on heuristics. |
| `GetPathFromRoot` | Gets the complete path of thoughts from root to this node as strings. |
| `IsLeaf` | Checks if this node is a leaf (has no children). |
| `IsRoot` | Checks if this node is the root (has no parent). |
| `ToString` | Returns a string representation of this thought node. |

