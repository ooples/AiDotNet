---
title: "ClassificationTaskType"
description: "Specifies the type of classification task being performed."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the type of classification task being performed.

## For Beginners

Classification is about putting things into categories.

Think of it like sorting mail:

- Binary: Is this spam or not spam? (2 categories)
- MultiClass: Is this a bill, letter, package, or advertisement? (multiple exclusive categories)
- MultiLabel: Mark all that apply: urgent, personal, work-related (multiple overlapping labels)
- Ordinal: Rate satisfaction: Poor, Fair, Good, Excellent (ordered categories)

The task type tells the model what kind of answer you're expecting.

## How It Works

Classification task types determine how the model interprets the target variable
and how predictions are structured. Different task types require different output
formats and loss functions.

## Fields

| Field | Summary |
|:-----|:--------|
| `Binary` | Binary classification with exactly two classes (e.g., spam/not-spam, positive/negative). |
| `MultiClass` | Multi-class classification where each sample belongs to exactly one of multiple classes. |
| `MultiLabel` | Multi-label classification where each sample can belong to multiple classes simultaneously. |
| `Ordinal` | Ordinal classification where classes have a natural ordering. |

