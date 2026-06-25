---
title: "SequentialSampler"
description: "A sampler that returns indices in sequential order without shuffling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Sampling`

A sampler that returns indices in sequential order without shuffling.

## For Beginners

This sampler goes through your data in order (0, 1, 2, 3, ...).
Use this when:

- Evaluating your model (you want consistent results)
- Making predictions on new data
- Debugging to isolate issues from randomness

## How It Works

SequentialSampler is useful during evaluation/inference when you want
deterministic, reproducible results without any randomness.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SequentialSampler(Int32)` | Initializes a new instance of the SequentialSampler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Length` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetIndicesCore` |  |

