---
title: "PyramidNER<T>"
description: "Pyramid-NER: Hierarchical pyramid network for nested Named Entity Recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.SpanBased`

Pyramid-NER: Hierarchical pyramid network for nested Named Entity Recognition.

## For Beginners

Imagine reading a sentence and first finding the smallest named
entities, then using those to help find bigger entities that contain them. For example,
first find "York" (location), then use that to find "New York" (location), then find
"New York University" (organization). Each step helps the next, building up like a pyramid.

## How It Works

Pyramid-NER (Jue et al., ACL 2020 - "Pyramid: A Layered Model for Nested Named Entity
Recognition") introduces a novel layered architecture where each pyramid layer identifies
entities at a specific nesting level, and inner entities become features for outer entities.

**Key Innovation - Layered Pyramid Architecture:**
Instead of treating nested NER as a single flat classification problem, Pyramid-NER
builds a pyramid of L layers, where:

- Layer 1: Identifies the innermost (shortest) entities
- Layer 2: Identifies entities that may contain Layer 1 entities
- Layer L: Identifies the outermost (longest) entities

Each layer uses a BiLSTM or transformer encoder, and the identified entities from
lower layers are fed as additional features to higher layers through "inverse pyramid"
connections.

**Architecture:****Inverse Pyramid Connections:**
When Layer k identifies an entity span (i, j), a "flag" embedding is generated and
concatenated with the token representations at layer k+1. This tells higher layers
"there is an entity of type X spanning positions i to j", enabling the model to
learn that outer entities often contain inner entities of specific types.

**Performance (Nested NER):**

- ACE 2004: ~86.1% F1
- ACE 2005: ~84.9% F1
- GENIA: ~78.5% F1
- NNE: ~93.7% F1

**Advantages:**

- Naturally handles arbitrarily deep nesting levels
- Lower layers provide useful features for higher layers
- Simple, interpretable architecture

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PyramidNER(NeuralNetworkArchitecture<>,SpanBasedNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Pyramid-NER model in native training mode. |
| `PyramidNER(NeuralNetworkArchitecture<>,String,SpanBasedNEROptions)` | Creates a Pyramid-NER model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateDefaultLayers` |  |
| `CreateNewInstance` |  |

