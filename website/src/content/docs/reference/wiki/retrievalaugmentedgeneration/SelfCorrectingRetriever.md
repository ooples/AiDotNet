---
title: "SelfCorrectingRetriever<T>"
description: "Self-correcting retriever that iteratively refines answers through critique, error detection, and targeted re-retrieval."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns`

Self-correcting retriever that iteratively refines answers through critique, error detection, and targeted re-retrieval.

## For Beginners

Think of this like writing an essay with self-editing.

Normal approach:

- Research topic once → Write essay → Submit (might have errors!)

Self-correcting approach:

- Research → Write draft → Read and critique → "Wait, I'm missing data about X"
- Research X specifically → Add to essay → Critique again → "This part contradicts that part"
- Research to resolve → Fix contradiction → Final review → Submit when satisfied

Example:
Question: "What caused the fall of the Roman Empire?"

Iteration 1:

- Retrieved: General docs about Roman Empire
- Answer: "Economic problems and barbarian invasions caused the fall"
- Critique: "Too vague - which economic problems? When did invasions happen?"
- Satisfied: NO

Iteration 2:

- Re-retrieve: "Roman Empire economic problems inflation"
- Answer: "Currency debasement and inflation in the 3rd century, plus Germanic invasions in 410 AD"
- Critique: "Better, but missing Eastern vs Western Empire distinction"
- Satisfied: NO

Iteration 3:

- Re-retrieve: "Western Roman Empire Eastern Byzantine"
- Answer: "Western Empire fell in 476 AD due to economics + invasions; Eastern continued as Byzantine"
- Critique: "Complete and accurate!"
- Satisfied: YES → Return answer

## How It Works

This advanced RAG pattern implements a self-correction loop: retrieve documents, generate answer,
critique the answer for errors or gaps, retrieve additional targeted documents, and repeat until
the answer is satisfactory. This mirrors how humans refine their understanding through iteration.

**How It Works:**
The self-correction process:

1. Initial Retrieval - Get top-K relevant documents for query
2. Generate Answer - Create initial answer from retrieved documents
3. Generate Critique - LLM critiques its own answer for errors/gaps
4. Check Satisfaction - Parse critique for approval keywords
5. If Not Satisfied:

a. Extract gaps - Identify what information is missing
b. Re-retrieve - Get documents about missing topics
c. Generate improved answer with all documents
d. Repeat critique (max 3 iterations)

6. Return final answer

Current implementation uses keyword detection for satisfaction.
Production should use structured critique (JSON) with explicit quality scores.

**Benefits:**

- Higher accuracy through iterative refinement
- Catches and corrects initial mistakes
- Identifies and fills knowledge gaps automatically
- More comprehensive answers
- Transparent - shows reasoning through critique

**Limitations:**

- Multiple LLM calls (higher cost/latency)
- May not converge if critique is inconsistent
- Depends heavily on LLM's self-critique ability
- Limited to max iterations (prevents infinite loops)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelfCorrectingRetriever(IGenerator<>,RetrieverBase<>,Int32)` | Initializes a new instance of the `SelfCorrectingRetriever` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `RetrieveAndAnswer(String,Int32,Dictionary<String,Object>)` | Retrieves documents and generates a self-corrected, refined answer through iterative critique. |

