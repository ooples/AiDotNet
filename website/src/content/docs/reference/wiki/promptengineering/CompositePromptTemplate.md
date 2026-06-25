---
title: "CompositePromptTemplate"
description: "Template that combines multiple prompt templates in sequence."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Templates`

Template that combines multiple prompt templates in sequence.

## For Beginners

Combines multiple templates into one.

Example:

## How It Works

This template allows chaining multiple templates together, making it easier
to build complex prompts from reusable components.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CompositePromptTemplate(IPromptTemplate[])` | Initializes a new instance of the CompositePromptTemplate class. |
| `CompositePromptTemplate(String,IPromptTemplate[])` | Initializes a new instance with a custom separator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputVariables` | Gets the list of all variable names from all component templates. |
| `Template` | Gets the raw template string (combined from all components). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(IPromptTemplate)` | Adds a template to the composite. |
| `Builder` | Creates a builder for constructing composite templates. |
| `Format(Dictionary<String,String>)` | Formats all templates and combines them. |
| `Validate(Dictionary<String,String>)` | Validates that all required variables are present. |

