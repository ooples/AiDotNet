---
title: "InitializationStrategy<T>"
description: "Provides backward-compatible access to initialization strategies."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Initialization`

Provides backward-compatible access to initialization strategies.

## How It Works

This class maintains backward compatibility with existing code that uses
`InitializationStrategy<T>.Lazy` etc.

## Properties

| Property | Summary |
|:-----|:--------|
| `Eager` | Gets the default eager initialization strategy. |
| `Lazy` | Gets the default lazy initialization strategy. |
| `Zero` | Gets the zero initialization strategy. |

