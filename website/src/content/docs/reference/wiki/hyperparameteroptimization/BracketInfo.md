---
title: "BracketInfo"
description: "Information about a Hyperband bracket."
section: "API Reference"
---

`Models & Types` · `AiDotNet.HyperparameterOptimization`

Information about a Hyperband bracket.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BracketInfo(Int32,Int32,Int32,List<ValueTuple<Int32,Int32>>)` | Initializes a new BracketInfo. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BracketIndex` | The bracket index (s value in Hyperband paper). |
| `InitialConfigurations` | Initial number of configurations in this bracket. |
| `InitialResource` | Initial resource budget per configuration. |
| `Rounds` | Rounds in this bracket with (configurations, resource) at each round. |
| `TotalResource` | Total resource units consumed by this bracket. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` |  |

