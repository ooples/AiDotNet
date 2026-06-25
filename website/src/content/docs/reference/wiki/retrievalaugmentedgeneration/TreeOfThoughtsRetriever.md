---
title: "TreeOfThoughtsRetriever<T>"
description: "Tree-of-Thoughts retriever that explores multiple reasoning paths in a tree structure."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns`

Tree-of-Thoughts retriever that explores multiple reasoning paths in a tree structure.

## For Beginners

Think of this like a chess player considering multiple moves.

Chain-of-Thought (linear):

- Question: "Impact of AI on healthcare?"
- Path: AI → Diagnosis → Treatment → Outcomes

Tree-of-Thoughts (branching):

- Question: "Impact of AI on healthcare?"
- Level 1: [AI in Diagnosis, AI in Treatment, AI in Research]
- Level 2 (from Diagnosis): [Image Analysis, Patient Records, Early Detection]
- Level 2 (from Treatment): [Drug Discovery, Personalized Medicine, Surgery Assistance]
- Explores all promising paths and selects best documents

This is especially useful when:

- Multiple valid reasoning approaches exist
- The problem requires exploring alternatives
- You want comprehensive coverage of a topic

## How It Works

This advanced retrieval pattern builds upon Chain-of-Thought by creating a tree of
possible reasoning paths. Instead of following a single linear chain, it explores
multiple branches of reasoning at each step, evaluates them, and can backtrack to
explore alternative paths. This enables more comprehensive exploration of complex
problem spaces.

**Example Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TreeOfThoughtsRetriever(IGenerator<>,RetrieverBase<>,Int32,Int32,TreeSearchStrategy,Int32)` | Initializes a new instance of the `TreeOfThoughtsRetriever` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildThoughtContext(ThoughtNode<>)` | Builds a context string from the parent chain of thoughts. |
| `CollectDocuments(ThoughtNode<>,Dictionary<String,ValueTuple<Document<>,Double>>)` | Collects all documents from the tree, keeping track of the best score for each. |
| `EvaluateAndRetrieve(ThoughtNode<>,Dictionary<String,Object>)` | Evaluates a thought node and retrieves relevant documents. |
| `EvaluateThought(ThoughtNode<>)` | Evaluates the quality of a thought based on retrieved documents and coherence. |
| `ExpandBestFirst(ThoughtNode<>,Dictionary<String,Object>)` | Best-first tree expansion: always explores the highest-scored node next. |
| `ExpandBreadthFirst(ThoughtNode<>,Dictionary<String,Object>)` | Breadth-first tree expansion: explores all nodes at each level. |
| `ExpandDepthFirst(ThoughtNode<>,Dictionary<String,Object>)` | Depth-first tree expansion: explores one branch fully before backtracking. |
| `ExpandTree(ThoughtNode<>,TreeSearchStrategy,Dictionary<String,Object>)` | Expands the reasoning tree using the specified search strategy. |
| `GenerateChildThoughts(ThoughtNode<>)` | Generates alternative child thoughts for a given node. |
| `ParseThoughts(String)` | Parses thoughts from LLM response. |
| `RetrieveCore(String,Int32,Dictionary<String,Object>)` | Core retrieval logic using tree-of-thoughts reasoning. |

