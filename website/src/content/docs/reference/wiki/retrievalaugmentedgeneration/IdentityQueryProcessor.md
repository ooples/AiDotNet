---
title: "IdentityQueryProcessor"
description: "A pass-through query processor that returns the query unchanged."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.QueryProcessors`

A pass-through query processor that returns the query unchanged.

## For Beginners

This processor does absolutely nothing to your query.

It's like asking someone to repeat what you just said:

- You: "What is the weather today?"
- Identity Processor: "What is the weather today?"

When to use this:

- You already have a well-formed query
- You want to disable query processing in your pipeline
- You're testing and want a baseline (no modifications)
- Your queries are coming from an API that pre-processes them

This is the simplest possible query processor - it just returns
exactly what you gave it, no changes at all.

## How It Works

This processor is useful as a default when no query processing is desired,
or as a baseline for testing and comparison.

## Methods

| Method | Summary |
|:-----|:--------|
| `ProcessQueryCore(String)` | Returns the query unchanged. |

