---
title: "NearMissVersion"
description: "Specifies the NearMiss version to use."
section: "API Reference"
---

`Enums` · `AiDotNet.Preprocessing.ImbalancedLearning`

Specifies the NearMiss version to use.

## For Beginners

- Version1: Keeps majority samples nearest to ANY minority sample
- Version2: Keeps majority samples nearest to the farthest minority sample
- Version3: Keeps k nearest majority neighbors for EACH minority sample

## Fields

| Field | Summary |
|:-----|:--------|
| `Version1` | Select majority samples based on distance to nearest minority samples. |
| `Version2` | Select majority samples based on distance to farthest minority samples. |
| `Version3` | Select k nearest majority neighbors for each minority sample. |

