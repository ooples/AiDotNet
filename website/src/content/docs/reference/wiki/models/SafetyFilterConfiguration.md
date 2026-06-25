---
title: "SafetyFilterConfiguration<T>"
description: "Configuration for safety filtering during inference."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration for safety filtering during inference.

## For Beginners

This is the safety "on/off switch" and settings bundle.
You can leave it alone to use safe defaults, or customize the filter for expert deployments.

## How It Works

This configuration controls whether safety filtering is enabled and which implementation is used.
When enabled and no custom filter is provided, a default filter is created using the provided options.

## Properties

| Property | Summary |
|:-----|:--------|
| `Enabled` | Gets or sets whether safety filtering is enabled. |
| `Filter` | Gets or sets an optional custom safety filter implementation. |
| `Options` | Gets or sets the default options used when constructing the standard safety filter. |

