---
title: "RELNER<T>"
description: "REL: Radboud Entity Linker - end-to-end entity linking combining NER with entity disambiguation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

REL: Radboud Entity Linker - end-to-end entity linking combining NER with entity disambiguation.

## For Beginners

REL is a complete system that first finds entity mentions in text
(NER step), then figures out which specific real-world entity each mention refers to
(entity linking step). For example, "Apple" could refer to Apple Inc. (the tech company)
or apple (the fruit). REL uses context clues and knowledge from Wikipedia to make the
right choice. It's like a three-step process: find mentions, generate candidates, pick
the best match.

## How It Works

REL (van Hulst et al., SIGIR 2020 - "REL: An Entity Linker Standing on the Shoulders of Giants")
is a modular end-to-end entity linking system that combines mention detection (NER),
candidate generation, and entity disambiguation into a single pipeline.

**Pipeline Architecture:****Stage 1 - Mention Detection (NER):**

- Uses Flair NER (BiLSTM-CRF with contextual string embeddings) for entity mention detection
- Identifies potential entity spans and their types (PER, ORG, LOC, MISC)
- Also supports n-gram-based mention detection for higher recall

**Stage 2 - Candidate Generation:**

- For each detected mention, generates candidate entities from a knowledge base (Wikipedia)
- Uses multiple signals: exact match, fuzzy match, prior probability P(entity|mention)
- Prior probabilities are computed from Wikipedia anchor text statistics
- Typically generates 30-100 candidates per mention

**Stage 3 - Entity Disambiguation (ED):**

- Uses a neural ED model to select the correct entity from candidates
- Features include:
- **Entity embeddings:** Pre-trained Wikipedia2Vec entity representations
- **Context similarity:** TF-IDF or neural similarity between mention context and

entity description

- **Coherence:** How well the candidate entity fits with other entities in the document

(collective entity linking)

- **Prior probability:** How likely this mention refers to this entity in general

**Collective Entity Linking:**
REL performs collective entity linking, where the disambiguation of one mention can
influence the disambiguation of others. For example, in "Jordan played basketball
for the Bulls", disambiguating "Bulls" as "Chicago Bulls" (sports team) helps
disambiguate "Jordan" as "Michael Jordan" (basketball player) rather than "Jordan"
(country).

**Performance:**

- AIDA-CoNLL: ~84.3% F1 (end-to-end, including NER)
- MSNBC: ~73.1% F1
- AQUAINT: ~82.4% F1
- Processing speed: ~40 documents/second on GPU

**API and Deployment:**
REL provides a REST API for easy deployment, making it practical for production use.
It supports both local and remote knowledge bases and can be extended with custom
entity catalogs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RELNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates a REL model in ONNX inference mode. |
| `RELNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a REL model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

