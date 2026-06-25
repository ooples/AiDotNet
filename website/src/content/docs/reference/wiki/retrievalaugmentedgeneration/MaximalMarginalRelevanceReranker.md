---
title: "MaximalMarginalRelevanceReranker<T>"
description: "Implements Maximal Marginal Relevance (MMR) reranking to balance relevance and diversity."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Rerankers`

Implements Maximal Marginal Relevance (MMR) reranking to balance relevance and diversity.

## For Beginners

MMR prevents search results from being too similar to each other.

The problem MMR solves:
Imagine searching for "climate change" and getting:

1. "Climate change threatens polar bears"
2. "Polar bears endangered by climate change"
3. "Climate change impact on polar ice affecting bears"
4. "Global warming threatens polar bear habitats"
5. "Arctic ice melting endangers polar bears"

All relevant, but they're all saying the same thing! You're getting one narrow aspect
repeated 5 times instead of a diverse view of climate change.

What MMR does instead:

1. "Climate change threatens polar bears" (relevant: ✓, diverse: ✓ first result)
2. "Rising sea levels threaten coastal cities" (relevant: ✓, different topic: ✓)
3. "Carbon emissions reach record highs" (relevant: ✓, different aspect: ✓)
4. "Renewable energy adoption accelerates globally" (relevant: ✓, solutions angle: ✓)
5. "Climate refugees increase in developing nations" (relevant: ✓, human impact: ✓)

Now you get a comprehensive view with diverse perspectives!

How MMR works:

1. Pick the most relevant document → Add to results
2. For next pick, consider:
- Relevance to query (you want relevant docs)
- Dissimilarity to already-picked docs (you want diversity)
3. Balance these two goals with a lambda parameter
4. Repeat until you have K documents

The lambda parameter (λ):

- λ = 1.0: Only care about relevance (normal ranking, no diversity)
- λ = 0.0: Only care about diversity (might get irrelevant but diverse docs)
- λ = 0.7: Balanced (70% relevance, 30% diversity) ← Good default

When to use MMR:

- Research/exploratory queries: Users want comprehensive coverage
- News aggregation: Don't show 10 articles about the same event
- Product search: Show variety, not just variations of one product
- Question answering: Provide multiple perspectives

When NOT to use MMR:

- User wants very specific info: "iPhone 15 Pro Max price" (diversity not helpful)
- Transactional queries: "buy Nike Air Max" (user knows what they want)
- Fact lookups: "Paris population" (one correct answer)

## How It Works

MMR reranking ensures that retrieved documents are not only relevant to the query but also
diverse from each other. This prevents redundancy where all top results say essentially the
same thing, providing users with a broader range of information. MMR is particularly valuable
for exploratory search, news aggregation, and research applications.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaximalMarginalRelevanceReranker(Func<Document<>,Vector<>>,Double)` | Initializes a new instance of the MaximalMarginalRelevanceReranker class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModifiesScores` | Gets a value indicating whether this reranker modifies relevance scores. |

## Methods

| Method | Summary |
|:-----|:--------|
| `RerankCore(String,IList<Document<>>)` | Reranks documents using Maximal Marginal Relevance. |

