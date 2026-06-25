---
title: "FederatedTextBenchmarkOptions"
description: "Configuration options for federated text benchmark suites."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for federated text benchmark suites.

## For Beginners

Text benchmarks test models on natural language problems. You select a suite (enum)
and provide the minimal dataset configuration here.

## How It Works

This groups dataset-specific options for text benchmarks (for example, Sent140 and Shakespeare) under a single
facade-facing configuration object.

## Properties

| Property | Summary |
|:-----|:--------|
| `Reddit` | Gets or sets Reddit options (LEAF Reddit JSON split files). |
| `Sent140` | Gets or sets Sent140 options (LEAF JSON split files). |
| `Shakespeare` | Gets or sets Shakespeare options (LEAF JSON split files). |
| `StackOverflow` | Gets or sets StackOverflow options (token sequence JSON split files). |

