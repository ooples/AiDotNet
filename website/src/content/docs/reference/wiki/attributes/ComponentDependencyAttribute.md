---
title: "ComponentDependencyAttribute"
description: "Declares a dependency that a component requires from another component or interface."
section: "API Reference"
---

`Attributes` · `AiDotNet.Attributes`

Declares a dependency that a component requires from another component or interface.

## For Beginners

This tells the pipeline builder what other components this one needs.
For example, a retriever might depend on an embedding model, or a reranker might depend on
a retriever. The pipeline builder uses this information to validate composition.

## How It Works

**Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ComponentDependencyAttribute(Type)` | Initializes a new instance of the `ComponentDependencyAttribute` class. |
| `ComponentDependencyAttribute(Type,String)` | Initializes a new instance of the `ComponentDependencyAttribute` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DependencyType` | Gets the type (typically an interface) that this component depends on. |
| `Description` | Gets or sets a description of why this dependency is needed. |
| `Required` | Gets or sets whether this dependency is required (true) or optional (false). |

