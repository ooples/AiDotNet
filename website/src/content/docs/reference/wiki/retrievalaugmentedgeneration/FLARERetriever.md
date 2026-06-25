---
title: "FLARERetriever<T>"
description: "FLARE (Forward-Looking Active REtrieval) pattern that actively decides when and what to retrieve during generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns`

FLARE (Forward-Looking Active REtrieval) pattern that actively decides when and what to retrieve during generation.

## For Beginners

Think of FLARE like asking follow-up questions when you're unsure.

Normal RAG:

- Question: "What is quantum computing?"
- Step 1: Retrieve all documents about quantum computing
- Step 2: Generate complete answer from those documents
- Problem: Might miss specific details or retrieve too much irrelevant info

FLARE:

- Question: "What is quantum computing?"
- Step 1: Start generating answer...
- Step 2: "Quantum computing uses quantum bits or..." (confident, keep going)
- Step 3: "...which leverage principles like..." (uncertain - what principles exactly?)
- Step 4: RETRIEVE more docs about "quantum principles superposition entanglement"
- Step 5: Continue with new information: "...superposition and entanglement..."
- Result: More focused retrieval, better coverage of uncertainty areas

It's like having a conversation where you ask for clarification only when you need it,
rather than reading an entire encyclopedia upfront.

## How It Works

FLARE (Forward-Looking Active REtrieval augmented generation) is an advanced RAG pattern that monitors
the language model's confidence during generation. When uncertainty is detected (low confidence on
next tokens), FLARE automatically retrieves additional relevant information to improve answer quality.
This creates a dynamic retrieval loop where retrieval happens only when needed, rather than all upfront.

**Example Usage:**

**How It Works:**
The retrieval process is:

1. Initial retrieval - Get top-3 relevant documents
2. Start generating answer with initial context
3. Monitor confidence - Check for uncertainty signals (keywords like "I'm not sure", "unclear")
4. Active retrieval - When uncertain, extract missing topics and retrieve more docs
5. Integrate new information - Continue generating with expanded context
6. Repeat - Maximum 5 iterations to prevent infinite loops
7. Return complete answer assembled from all iterations

Current implementation uses keyword detection for uncertainty. Production version would use:

- Token-level confidence scores (logprobs from LLM)
- Attention weights to identify knowledge gaps
- Explicit uncertainty statements from the model

**Benefits:**

- More efficient retrieval - Only fetches what's needed
- Better coverage - Addresses uncertainty areas specifically
- Reduced noise - Avoids retrieving irrelevant documents upfront
- Adaptive - Responds to complexity of the question dynamically
- Cost-effective - Fewer total documents retrieved vs exhaustive upfront retrieval

**Limitations:**

- Requires LLM with confidence scores (logprobs) for best results
- Multiple LLM calls increase latency
- May miss information if uncertainty detection fails
- Current implementation uses simple keyword matching (needs improvement with real LLM logprobs)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FLARERetriever(IGenerator<>,RetrieverBase<>,Double)` | Initializes a new instance of the `FLARERetriever` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateWithActiveRetrieval(String)` | Generates an answer with active retrieval triggered by detected uncertainty. |

