---
title: "PromptNER<T>"
description: "PromptNER: Prompt-based learning for few-shot Named Entity Recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

PromptNER: Prompt-based learning for few-shot Named Entity Recognition.

## For Beginners

PromptNER uses descriptions of entity types (like "a person's name")
instead of simple labels (like "PER") to help the model understand what it's looking for.
This makes it much easier to adapt to new entity types - just write a description of what
the new type looks like, and the model can start recognizing it without extensive retraining.

## How It Works

PromptNER (Shen et al., EMNLP 2023 - "PromptNER: Prompt Locating and Typing for Named
Entity Recognition") uses prompt-based learning with entity type descriptions to enable
effective few-shot NER. It combines the advantages of prompt tuning with span-level
entity recognition.

**Key Innovation - Entity Type Descriptions as Prompts:**
Instead of using class labels like "PER", "ORG", "LOC", PromptNER uses rich natural
language descriptions of entity types as prompts:

- PER: "a person's name, including first names, last names, and full names"
- ORG: "the name of an organization, company, institution, or team"
- LOC: "a geographical location, city, country, landmark, or region"

These descriptions are encoded by the model and used to compute similarity scores
between candidate entity spans and entity type descriptions.

**Architecture:**

1. **Text Encoder:** Encode the input sentence with a pre-trained transformer
2. **Prompt Encoder:** Encode each entity type description with the same transformer
3. **Span Locating:** Identify candidate entity spans using learned span boundaries
4. **Span Typing:** Compute similarity between each candidate span representation

and each entity type prompt representation

5. **Label Assignment:** Assign each span the entity type with the highest similarity

**Few-Shot Learning Mechanism:**

- The prompt descriptions act as "soft labels" that carry semantic information
- Adding a few labeled examples refines the span boundary detection
- The similarity-based typing naturally generalizes to new entity types

**Performance:**

- 5-shot CoNLL-2003: ~68-73% F1
- 20-shot CoNLL-2003: ~78-83% F1
- Full training CoNLL-2003: ~93.2% F1
- Zero-shot transfer to new domains: ~55-65% F1

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PromptNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates a PromptNER model in ONNX inference mode. |
| `PromptNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a PromptNER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

