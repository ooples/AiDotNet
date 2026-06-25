---
title: "GeneratorBase<T>"
description: "Base class for generator implementations providing common functionality and validation."
section: "API Reference"
---

`Base Classes` · `AiDotNet.RetrievalAugmentedGeneration.Generators`

Base class for generator implementations providing common functionality and validation.

## For Beginners

This is the foundation for all text generators in RAG systems.

It handles common tasks so you don't have to repeat them:

- Checking that inputs aren't null or empty
- Building prompts that combine the query and retrieved documents
- Extracting citations from generated text
- Creating properly formatted answers

When you create a new generator (like OpenAIGenerator or OnnxGenerator):

1. Inherit from this class
2. Set MaxContextTokens and MaxGenerationTokens in the constructor
3. Implement GenerateCore with your specific generation logic
4. Everything else (validation, prompt formatting, citations) is handled automatically

## How It Works

This base class provides standard validation, prompt construction, and citation handling
for generator implementations. It defines the template for implementing custom generation logic.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GeneratorBase(Int32,Int32)` | Initializes a new instance of the GeneratorBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxContextTokens` | Gets the maximum number of tokens this generator can process in a single request. |
| `MaxGenerationTokens` | Gets the maximum number of tokens this generator can generate in a response. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildPromptWithContext(String,List<Document<>>)` | Builds a prompt that incorporates the query and retrieved context documents. |
| `ExtractCitations(String,List<Document<>>)` | Extracts citation markers from the generated text and maps them to source documents. |
| `Generate(String)` | Generates a text response based on a prompt with validation. |
| `GenerateCore(String)` | Core generation logic to be implemented by derived classes. |
| `GenerateGrounded(String,IEnumerable<Document<>>)` | Generates a grounded answer using provided context documents. |

