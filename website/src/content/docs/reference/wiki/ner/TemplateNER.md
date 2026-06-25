---
title: "TemplateNER<T>"
description: "Template-NER: Template-based prompt approach for few-shot and zero-shot NER."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

Template-NER: Template-based prompt approach for few-shot and zero-shot NER.

## For Beginners

Template-NER treats entity recognition like a fill-in-the-blank exercise.
Instead of labeling each word, it asks the model: "What person is mentioned in this sentence?"
and the model answers: "Barack Obama." This approach works even with very few training examples
because the model already understands language from pre-training.

## How It Works

Template-NER (Cui et al., ACL 2021 - "Template-Based Named Entity Recognition Using BART")
reformulates NER as a sequence-to-sequence generation problem using natural language templates.
Instead of predicting labels, the model generates text that fills in template slots.

**Key Innovation - Template-Based Generation:**
Given an input sentence, Template-NER constructs a template that describes the NER task
in natural language, then uses a seq2seq model (BART) to generate the filled template.

**Example:**
Input: "Barack Obama visited New York City"
Template: "[Person] visited [Location]"
Model generates: "Barack Obama visited New York City"
Extracted: Person="Barack Obama", Location="New York City"

**How Templates Work:**

1. Define templates for each entity type: "[Person] is a person", "[Location] is a location"
2. For each entity type, prompt the model: "In the sentence X, [Person] is a person"
3. The model generates the entity that fills [Person] based on the sentence context
4. Repeat for all entity types to extract all entities

**Few-Shot Capability:**
Because the templates are expressed in natural language, the model can leverage its
pre-trained knowledge to perform NER with very few labeled examples:

- Zero-shot: No labeled data, just template definitions
- Few-shot (10 examples): ~70-75% F1 on CoNLL-2003
- Few-shot (100 examples): ~80-85% F1 on CoNLL-2003
- Full training: ~93.0% F1 on CoNLL-2003

**Template Design Patterns:**

- Entity extraction: "In the sentence, [TYPE] refers to ___"
- Type classification: "___ is a [PERSON/LOCATION/ORGANIZATION]"
- Cloze-style: "Obama is a [MASK] entity" -> "person"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TemplateNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates a Template-NER model in ONNX inference mode. |
| `TemplateNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Template-NER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

