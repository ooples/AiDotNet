---
title: "PromptTemplateFactory"
description: "Factory for creating prompt template instances based on template type."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Factories`

Factory for creating prompt template instances based on template type.

## For Beginners

A factory that creates the right kind of template for you.

Example:
```cs
// Create a simple template
var simpleTemplate = PromptTemplateFactory.Create(
PromptTemplateType.Simple,
"Translate {text} to {language}"
);

// Create a chat template
var chatTemplate = PromptTemplateFactory.Create(
PromptTemplateType.Chat
) as ChatPromptTemplate;
chatTemplate.AddSystemMessage("You are a helpful assistant");
```

## How It Works

This factory creates prompt templates based on the specified type, following the
factory pattern used throughout the AiDotNet library.

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(PromptTemplateType,String,IFewShotExampleSelector<>,Int32)` | Creates a prompt template of the specified type with a strongly-typed few-shot example selector. |
| `Create(PromptTemplateType,String,Int32)` | Creates a prompt template of the specified type. |
| `CreateChainOfThoughtTemplate(String)` | Creates a chain-of-thought prompt template. |
| `CreateChatTemplate` | Creates a chat prompt template. |
| `CreateFewShotTemplate(String,IFewShotExampleSelector<>,Int32)` | Creates a few-shot prompt template. |
| `CreateReActTemplate(String)` | Creates a ReAct (Reasoning + Acting) prompt template. |
| `CreateSimpleTemplate(String)` | Creates a simple prompt template. |

