---
title: "FineTuningCategory"
description: "Categories of fine-tuning methods."
section: "API Reference"
---

`Enums` · `AiDotNet.Interfaces`

Categories of fine-tuning methods.

## For Beginners

These categories group fine-tuning methods by how they learn.
Some learn from labeled data, others from preferences, and some from reward signals.

## Fields

| Field | Summary |
|:-----|:--------|
| `Adversarial` | Adversarial methods that use game-theoretic approaches. |
| `Constitutional` | Constitutional AI methods that use principles for self-improvement. |
| `Contrastive` | Contrastive methods that learn from positive/negative examples. |
| `DirectPreference` | Direct Preference Optimization - learns directly from preference pairs. |
| `KnowledgeDistillation` | Knowledge distillation - transfer knowledge from teacher to student. |
| `OddsRatioPreference` | Odds/Ratio-based methods that combine SFT and preference learning. |
| `RankingBased` | Ranking-based methods that learn from response rankings. |
| `ReinforcementLearning` | Reinforcement Learning - learns from reward signals. |
| `SelfPlay` | Self-play methods where the model learns from itself. |
| `SupervisedFineTuning` | Supervised Fine-Tuning - learns from labeled input-output pairs. |

