---
title: "Compose<T>"
description: "Chains multiple transforms of the same type into a single transform."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Transforms`

Chains multiple transforms of the same type into a single transform.

## For Beginners

Compose lets you combine multiple transforms into one.
Instead of applying each transform separately, you create a pipeline:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Compose(IEnumerable<ITransform<,>>)` | Creates a composed transform from an enumerable of transforms. |
| `Compose(ITransform<,>[])` | Creates a composed transform from an ordered list of transforms. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the number of transforms in this composition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply()` |  |

