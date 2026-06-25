---
title: "PromptChain"
description: "Composes multiple prompt templates into a single formatted prompt."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering`

Composes multiple prompt templates into a single formatted prompt.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PromptChain(String)` | Initializes a new prompt chain. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Templates` | Gets the templates in this chain. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(IPromptTemplate)` | Adds a prompt template to the chain. |
| `Format(Dictionary<String,String>)` | Formats all templates in insertion order and joins the outputs. |

