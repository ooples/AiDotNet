---
title: "W2NER<T>"
description: "W2NER: Word-Word Relation Classification for unified flat and nested NER."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.SpanBased`

W2NER: Word-Word Relation Classification for unified flat and nested NER.

## For Beginners

W2NER looks at every pair of words in a sentence and asks: "Are these
two words part of the same entity?" Instead of labeling each word separately, it builds a
grid showing relationships between all word pairs. This is powerful because it can handle
normal entities, nested entities, and even discontinuous entities (where an entity has
gaps) using the same unified approach.

## How It Works

W2NER (Li et al., AAAI 2022 - "Unified Named Entity Recognition as Word-Word Relation
Classification") reformulates NER as a word-word relation classification problem, where
the relation between every pair of words indicates whether they are part of the same entity
and what type of entity they form.

**Key Innovation - Word-Word Relations:**
Instead of labeling individual tokens (BIO) or classifying spans, W2NER builds an n x n
table (where n is sentence length) and classifies each word-word pair into one of:

- **None:** Words are not related by any entity
- **Next-Neighboring-Word (NNW):** Words are consecutive within the same entity
- **Tail-Head-Word-* (THW-TYPE):** The pair represents the (tail, head) boundary

of an entity of the given TYPE

**Example:**
Sentence: "Barack Obama visited New York City"

- (Barack, Obama) = NNW (consecutive in same entity)
- (Obama, Barack) = THW-PER (tail-head pair of PER entity "Barack Obama")
- (New, York) = NNW, (York, City) = NNW
- (City, New) = THW-LOC (tail-head pair of LOC entity "New York City")

**Architecture:**

1. **BERT Encoder:** Produces contextual token representations
2. **Convolution Layer:** A convolutional layer over the word-pair grid captures

local interactions between neighboring word pairs

3. **Co-Predictor:** Combines token-level and grid-level features using:
- CLN (Conditional Layer Normalization): h_ij conditioned on h_i and h_j
- Distance embeddings: relative position between word pairs
- Biaffine scoring for the final word-word relation classification

**Advantages:**

- Unified framework: handles flat, nested, and discontinuous entities
- No span enumeration: avoids the O(n^2) span enumeration cost
- Grid structure captures inter-entity dependencies naturally

**Performance:**

- CoNLL-2003 (flat): ~93.4% F1
- ACE 2004 (nested): ~87.3% F1
- ACE 2005 (nested): ~86.6% F1
- GENIA (nested): ~79.8% F1

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `W2NER(NeuralNetworkArchitecture<>,SpanBasedNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a W2NER model in native training mode. |
| `W2NER(NeuralNetworkArchitecture<>,String,SpanBasedNEROptions)` | Creates a W2NER model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateDefaultLayers` |  |
| `CreateNewInstance` |  |

