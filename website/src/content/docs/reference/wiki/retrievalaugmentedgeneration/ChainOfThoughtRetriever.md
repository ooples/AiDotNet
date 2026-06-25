---
title: "ChainOfThoughtRetriever<T>"
description: "Chain-of-Thought retriever that generates reasoning steps before retrieving documents."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns`

Chain-of-Thought retriever that generates reasoning steps before retrieving documents.

## For Beginners

Think of this like asking a research assistant to explain their thought process.

Normal retriever:

- Question: "How does photosynthesis impact climate change?"
- Action: Search for documents about "photosynthesis" and "climate change"

Chain-of-Thought retriever:

- Question: "How does photosynthesis impact climate change?"
- Reasoning: "First, I need to understand what photosynthesis is. Then, I need to know how it

relates to carbon dioxide. Finally, I need to connect CO2 to climate change."

- Actions:
1. Search for "what is photosynthesis"
2. Search for "photosynthesis carbon dioxide absorption"
3. Search for "CO2 levels and climate change"
- Result: More complete answer because we gathered all prerequisite knowledge

This is especially useful for complex questions that require understanding multiple concepts
in a specific order.

## How It Works

This advanced retrieval pattern uses large language models to break down complex queries
into intermediate reasoning steps before retrieving documents. By generating a chain of
thought, the retriever can identify key concepts, sub-questions, and the logical order
in which information should be gathered, leading to more comprehensive and relevant results.

**Example Usage:**

**Production Readiness:**
Current implementation uses IGenerator interface which can accept:

- StubGenerator for development/testing
- Real LLM (GPT-4, Claude, Gemini) for production

To make production-ready:

1. Replace StubGenerator with real LLM generator
2. Optionally tune the reasoning prompt for your domain
3. Adjust max sub-queries limit based on LLM costs
4. Consider caching reasoning for common queries

**Benefits:**

- More comprehensive results for complex queries
- Better coverage of prerequisite knowledge
- Improved relevance through structured reasoning
- Transparent reasoning process for debugging
- Self-consistency improves robustness

**Limitations:**

- Requires LLM access (costs/latency)
- Quality depends on LLM reasoning ability
- May retrieve redundant documents if reasoning overlaps
- Slower than direct retrieval

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChainOfThoughtRetriever(IGenerator<>,RetrieverBase<>,List<String>)` | Initializes a new instance of the `ChainOfThoughtRetriever` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildReasoningPrompt(String,Int32)` | Builds a reasoning prompt with optional few-shot examples and variation for self-consistency. |
| `ExtractSubQueries(String,String)` | Extracts sub-queries from LLM reasoning with production-ready fuzzy deduplication. |
| `IsDuplicate(String,List<String>)` | Checks if a normalized query is a fuzzy duplicate of existing queries. |
| `JaroWinklerSimilarity(String,String)` | Computes Jaro-Winkler similarity between two strings (0.0 to 1.0, where 1.0 is identical). |
| `NormalizeQuery(String)` | Normalizes a query string for better matching and deduplication. |
| `Retrieve(String,Int32)` | Retrieves documents using chain-of-thought reasoning. |
| `Retrieve(String,Int32,Dictionary<String,Object>)` | Retrieves documents using chain-of-thought reasoning with metadata filtering. |
| `RetrieveWithSelfConsistency(String,Int32,Int32,Dictionary<String,Object>)` | Retrieves documents using self-consistency chain-of-thought reasoning. |

