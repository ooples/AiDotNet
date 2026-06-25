---
title: "Pruning"
description: "All 6 public types in the AiDotNet.pruning namespace, organized by kind."
section: "API Reference"
---

**6** public types in this namespace, organized by kind.

## Models & Types (5)

| Type | Summary |
|:-----|:--------|
| [`GradientPruningStrategy<T>`](/docs/reference/wiki/pruning/gradientpruningstrategy/) | Prunes weights based on gradient magnitude (sensitivity). |
| [`LotteryTicketPruningStrategy<T>`](/docs/reference/wiki/pruning/lotteryticketpruningstrategy/) | Implements the Lottery Ticket Hypothesis (Frankle & Carbin, 2019). |
| [`MagnitudePruningStrategy<T>`](/docs/reference/wiki/pruning/magnitudepruningstrategy/) | Prunes weights with smallest absolute values. |
| [`PruningMask<T>`](/docs/reference/wiki/pruning/pruningmask/) | Binary mask for pruning neural network weights. |
| [`StructuredPruningStrategy<T>`](/docs/reference/wiki/pruning/structuredpruningstrategy/) | Structured pruning removes entire neurons/filters/channels. |

## Enums (1)

| Type | Summary |
|:-----|:--------|
| [`StructurePruningType<T>`](/docs/reference/wiki/pruning/structurepruningtype/) | Defines the type of structural unit to prune. |

