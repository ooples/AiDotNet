---
title: "ModerationVerdict"
description: "The verdict from moderating a piece of content: whether it is allowed, and if not, why."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Pipeline`

The verdict from moderating a piece of content: whether it is allowed, and if not, why.

## Properties

| Property | Summary |
|:-----|:--------|
| `Allowed` | Gets a value indicating whether the content is permitted. |
| `Reason` | Gets the reason the content was blocked, or `null` when allowed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Allow` | A verdict permitting the content. |
| `Block(String)` | A verdict blocking the content with a reason. |

