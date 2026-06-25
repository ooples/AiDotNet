---
title: "JMPOptions<T, TInput, TOutput>"
description: "Configuration options for JMP (Joint Multi-Phase meta-learning)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for JMP (Joint Multi-Phase meta-learning).

## How It Works

JMP uses a multi-phase inner loop with separate learning rates and regularization.
Phase 1 (coarse) uses a higher learning rate for fast, rough adaptation.
Phase 2 (fine) uses a lower learning rate with stronger regularization toward the
Phase 1 result for careful refinement.

## Properties

| Property | Summary |
|:-----|:--------|
| `Phase1Fraction` | Fraction of adaptation steps in Phase 1 (coarse). |
| `Phase1LRMultiplier` | Learning rate multiplier for Phase 1 (coarse). |
| `Phase2LRMultiplier` | Learning rate multiplier for Phase 2 (fine). |
| `PhaseRegWeight` | L2 regularization during Phase 2 toward Phase 1 result. |

