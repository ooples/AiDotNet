---
title: "InferenceScopeHandle<T>"
description: "A disposable handle that restores the previous inference context when disposed."
section: "API Reference"
---

`Structs` · `AiDotNet.Memory`

A disposable handle that restores the previous inference context when disposed.
Returned by `InferenceContext{` to enable proper scope nesting.

## How It Works

This struct should be used with a using statement to ensure the previous context
is properly restored even if an exception occurs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InferenceScopeHandle(InferenceContext<>)` | Initializes a new handle that will restore the specified context on disposal. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` | Restores the previous inference context. |

