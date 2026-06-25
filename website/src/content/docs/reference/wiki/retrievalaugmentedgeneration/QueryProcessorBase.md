---
title: "QueryProcessorBase"
description: "Base class for query processor implementations with common validation logic."
section: "API Reference"
---

`Base Classes` · `AiDotNet.RetrievalAugmentedGeneration.QueryProcessors`

Base class for query processor implementations with common validation logic.

## For Beginners

This is the foundation for all query processors.

It handles the boring stuff so you can focus on the interesting parts:

- Checking that the query isn't empty or null
- Providing a clean structure for your processing logic
- Ensuring consistent error handling

When you create a new query processor, you just need to:

1. Inherit from this class
2. Implement ProcessQueryCore with your custom logic
3. Everything else is handled for you

## How It Works

This base class provides standard validation for query processors and defines
the template for implementing custom query processing logic.

## Methods

| Method | Summary |
|:-----|:--------|
| `ProcessQuery(String)` | Processes the query with validation. |
| `ProcessQueryCore(String)` | Core query processing logic to be implemented by derived classes. |

