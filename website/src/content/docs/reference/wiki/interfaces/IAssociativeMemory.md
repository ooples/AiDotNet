---
title: "IAssociativeMemory<T>"
description: "Interface for Associative Memory modules used in nested learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for Associative Memory modules used in nested learning.
Models both backpropagation and attention mechanisms as associative memory.

## Properties

| Property | Summary |
|:-----|:--------|
| `Capacity` | Gets the memory capacity. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Associate(Vector<>,Vector<>)` | Associates an input with a target output (learns the mapping). |
| `Clear` | Clears all stored associations. |
| `Retrieve(Vector<>)` | Retrieves the associated output for a given input query. |
| `Update(Vector<>,Vector<>,)` | Updates the memory based on new associations. |

