---
title: "VerifiedReasoningRetriever<T>"
description: "Verified reasoning retriever that validates each reasoning step with critic models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns`

Verified reasoning retriever that validates each reasoning step with critic models.

## For Beginners

Think of this like having a fact-checker review each step
of your reasoning.

Regular Chain-of-Thought:

- Generate reasoning steps
- Retrieve documents
- Return results

Verified Reasoning:

- Generate a reasoning step
- Retrieve supporting documents
- Ask a critic: "Is this step well-supported by the documents?"
- If not, refine the step or try a different approach
- Continue only with verified steps

This is useful when:

- Accuracy is critical (medical, legal, scientific domains)
- You want to avoid hallucinations or unsupported claims
- You need transparent, verifiable reasoning chains

## How It Works

This advanced retrieval pattern adds verification and self-refinement to chain-of-thought
reasoning. Each reasoning step is evaluated by a critic model to ensure it's well-supported
by retrieved evidence. If a step is found to be weak or unsupported, the system can
refine it or generate alternative reasoning paths.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VerifiedReasoningRetriever(IGenerator<>,RetrieverBase<>,Double,Int32,Int32)` | Initializes a new instance of the `VerifiedReasoningRetriever` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateReasoningChain(String)` | Generates initial reasoning chain for the query. |
| `ParseCriticResponse(String)` | Parses critic response to extract score and feedback. |
| `ParseReasoningSteps(String)` | Parses reasoning steps from LLM response. |
| `RefineStep(String,String,String)` | Refines a reasoning step based on critic feedback. |
| `RetrieveCore(String,Int32,Dictionary<String,Object>)` | Core retrieval logic using verified reasoning. |
| `RetrieveWithVerification(String,Int32,Dictionary<String,Object>)` | Retrieves documents using verified reasoning with critic feedback. |
| `VerifyAndRefineStep(String,String,Dictionary<String,Object>)` | Verifies a reasoning step and refines it if necessary. |
| `VerifyStep(String,List<Document<>>,String)` | Verifies a reasoning step using a critic model. |

