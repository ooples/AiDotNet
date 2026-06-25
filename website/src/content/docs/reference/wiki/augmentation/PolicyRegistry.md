---
title: "PolicyRegistry<T>"
description: "Registry for built-in and custom augmentation policies."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Augmentation.Image`

Registry for built-in and custom augmentation policies.

## Methods

| Method | Summary |
|:-----|:--------|
| `Get(String)` | Gets a policy by name. |
| `GetNames` | Gets all registered policy names. |
| `Register(String,Func<AugmentationPolicy<>>)` | Registers a named policy factory. |

