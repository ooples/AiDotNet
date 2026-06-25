---
title: "CurriculumScheduleType"
description: "Types of curriculum scheduling strategies."
section: "API Reference"
---

`Enums` · `AiDotNet.CurriculumLearning.Interfaces`

Types of curriculum scheduling strategies.

## Fields

| Field | Summary |
|:-----|:--------|
| `BabySteps` | Baby steps - very gradual introduction of harder samples. |
| `CompetenceBased` | Competence-based - advances when model masters content. |
| `Cosine` | Cosine annealing progression. |
| `Exponential` | Exponential increase in data fraction. |
| `Linear` | Linear increase in data fraction over epochs. |
| `Logarithmic` | Logarithmic growth (fast initial increase, then slow). |
| `OnePass` | One-pass - each sample seen exactly once in curriculum order. |
| `Polynomial` | Polynomial curve progression. |
| `SelfPaced` | Self-paced learning (SPL) - adapts based on sample losses. |
| `Step` | Fixed step increases at regular intervals. |

