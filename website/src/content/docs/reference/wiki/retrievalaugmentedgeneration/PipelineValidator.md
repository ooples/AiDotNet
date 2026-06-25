---
title: "PipelineValidator"
description: "Validates that a set of components forms a valid AI pipeline at runtime."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.RetrievalAugmentedGeneration.Configuration`

Validates that a set of components forms a valid AI pipeline at runtime.

## For Beginners

Think of this like a pre-flight checklist for your AI pipeline.

Before an airplane takes off, pilots check that all systems are working:

- Engines? Check.
- Navigation? Check.
- Fuel? Check.

Similarly, before running a RAG pipeline, this validator checks:

- Do you have a retriever? Check.
- Do you have a generator? Check.
- Are components in the right stages? Check.

If something is missing or misconfigured, you get a clear error message
instead of a confusing crash at runtime.

## How It Works

PipelineValidator provides static methods to check whether a collection of components
can form a valid pipeline before execution. It catches common misconfigurations early
with clear error messages, preventing cryptic runtime failures.

## Methods

| Method | Summary |
|:-----|:--------|
| `ValidatePipeline(IReadOnlyList<ValueTuple<ComponentType,PipelineStage>>)` | Validates a generic pipeline has no conflicting or missing stages. |
| `ValidateRAGConfiguration(Boolean,Boolean,Boolean,Boolean,Boolean,Boolean,Boolean)` | Validates the RAG components that have been configured on the builder. |
| `ValidateRAGPipeline(IReadOnlyList<PipelineStage>,IReadOnlyList<ComponentType>)` | Validates a RAG pipeline has all required stages and compatible components. |

