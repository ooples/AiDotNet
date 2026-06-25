---
title: "SimCSEType"
description: "Defines the training paradigms for SimCSE (Simple Contrastive Learning of Sentence Embeddings)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the training paradigms for SimCSE (Simple Contrastive Learning of Sentence Embeddings).

## For Beginners

Think of this as the "learning style" of the model. 

- **Unsupervised** is like a student learning by comparing different versions of the same book.
- **Supervised** is like a student learning from a teacher who provides "true" and "false" examples.

## How It Works

SimCSE supports two primary modes:

- **Unsupervised:** Uses dropout masks on identical sentence pairs as a minimal data augmentation.
- **Supervised:** Uses labeled entailment and contradiction pairs from datasets like SNLI or MultiNLI.

## Fields

| Field | Summary |
|:-----|:--------|
| `Supervised` | Supervised learning using Natural Language Inference (NLI) datasets. |
| `Unsupervised` | Unsupervised learning using dropout as noise. |

