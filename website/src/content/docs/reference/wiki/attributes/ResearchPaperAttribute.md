---
title: "ResearchPaperAttribute"
description: "Specifies the academic paper(s) that introduced or describe a model, component, or algorithm."
section: "API Reference"
---

`Attributes` · `AiDotNet.Attributes`

Specifies the academic paper(s) that introduced or describe a model, component, or algorithm.

## For Beginners

Apply this attribute to any class to reference the research paper
that describes how it works. This gives users a way to understand the theory and verify
correctness. You can apply it multiple times for classes based on multiple papers.

## How It Works

This attribute works for all three metadata tiers:

- Tier 1 (Models): "Attention Is All You Need" on Transformer
- Tier 2 (Components): "ColBERT: Efficient and Effective Passage Search" on ColBERTRetriever
- Tier 3 (Infrastructure): "Billion-scale similarity search with GPUs" on FaissIndex

**Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ResearchPaperAttribute(String,String)` | Initializes a new instance of the `ResearchPaperAttribute` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Authors` | Gets or sets the authors of the paper. |
| `Title` | Gets the title of the paper. |
| `Url` | Gets the URL where the paper can be accessed (typically an arXiv or DOI link). |
| `Year` | Gets or sets the year the paper was published. |

