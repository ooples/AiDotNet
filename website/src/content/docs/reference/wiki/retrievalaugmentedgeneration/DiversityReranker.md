---
title: "DiversityReranker<T>"
description: "Reranks documents to maximize diversity while maintaining relevance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Rerankers`

Reranks documents to maximize diversity while maintaining relevance.

## For Beginners

This prevents showing the same information multiple times.

The Problem:
Imagine searching for "Python programming" and getting 10 results:

- Result 1: "Python is a programming language..."
- Result 2: "Python is a programming language used for..."
- Result 3: "Python programming language allows..."
- Result 4-10: More variations of the same thing

That's redundant! You want variety:

- Result 1: Python basics
- Result 2: Python web development
- Result 3: Python data science
- Result 4: Python machine learning
- Result 5: Python performance tips

How it works:

1. Pick the most relevant document first
2. For remaining docs, balance two factors:

a) Relevance to the query (should be useful)
b) Difference from already-picked docs (should be unique)

3. Keep picking until you have enough results

Diversity calculation:

- Compares text overlap (how many words are shared)
- Higher overlap = less diverse = lower score
- Lower overlap = more diverse = higher score

Lambda parameter (0 to 1):

- lambda=1.0: Only care about relevance (might get duplicates)
- lambda=0.0: Only care about diversity (might get irrelevant docs)
- lambda=0.5: Balance both (recommended default)

Real example with lambda=0.5:
Query: "climate change effects"

Step 1: Pick most relevant → "Climate change causes rising temperatures" (relevance: 0.9)
Step 2: Next candidates:

- "Climate change leads to warmer weather" (relevance: 0.85, similarity to picked: 0.7)

→ Score: 0.5 * 0.85 - 0.5 * 0.7 = 0.075

- "Ocean acidification from CO2" (relevance: 0.7, similarity: 0.2)

→ Score: 0.5 * 0.7 - 0.5 * 0.2 = 0.25 ✓ Pick this!

Result: You get coverage of temperature AND ocean effects, not just temperature twice!

When to use this:

- Search results where redundancy is common
- Document recommendation systems
- Exploratory searches where breadth matters
- After initial retrieval that returns many similar docs

## How It Works

This reranker addresses the problem of redundant results by explicitly promoting diversity.
It uses a greedy algorithm to select documents that are both relevant to the query and
dissimilar from already-selected documents. This is similar to Maximal Marginal Relevance (MMR)
but uses a simpler diversity metric based on text overlap.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiversityReranker()` | Initializes a new instance of the DiversityReranker class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModifiesScores` | Gets a value indicating whether this reranker modifies relevance scores. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateTextSimilarity(String,String)` | Calculates text similarity based on word overlap (Jaccard similarity of word sets). |
| `RerankCore(String,IList<Document<>>)` | Core reranking logic that maximizes diversity while maintaining relevance. |

