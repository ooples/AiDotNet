---
title: "TriaffineNER<T>"
description: "Triaffine-NER: Three-way interaction model for nested Named Entity Recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.SpanBased`

Triaffine-NER: Three-way interaction model for nested Named Entity Recognition.

## For Beginners

Think of Triaffine-NER as an improved version of Biaffine-NER.
While Biaffine only looks at the first and last word of a potential entity, Triaffine
also considers what's in the middle. This helps distinguish entities that start and end
the same way but have different content, like "Bank of America" vs "Bank of the River."

## How It Works

Triaffine-NER (Yuan et al., ACL 2022 - "Fusing Heterogeneous Factors with Triaffine
Mechanism for Nested Named Entity Recognition") extends biaffine scoring with a third
factor that captures span content information, enabling richer span representations
for nested NER.

**Key Innovation - Triaffine Mechanism:**
While Biaffine-NER scores spans using only start and end boundary tokens:
score_biaffine(i,j) = h_start_i^T * W * h_end_j

Triaffine-NER adds a third factor that represents the span content:
score_triaffine(i,j) = h_start_i^T * W(h_content_{i:j}) * h_end_j

where W(h_content) is a weight matrix conditioned on the span content representation.
This creates a three-way interaction between start boundary, end boundary, and content,
allowing the model to differentiate spans with similar boundaries but different content.

**Heterogeneous Factors:**
The three factors capture different aspects of entity spans:

1. **Start boundary (h_start):** Left context and entity beginning patterns
2. **End boundary (h_end):** Right context and entity ending patterns
3. **Content (h_content):** Internal span semantics (pooled over span tokens)

**Architecture:**

1. Pre-trained transformer encoder produces token representations
2. Three separate MLPs transform tokens into start, end, and content representations
3. Triaffine scoring: For each (start, end, content) triple, compute entity type scores
4. Greedy or optimal decoding to extract non-conflicting entity spans

**Performance (Nested NER):**

- ACE 2004: ~87.8% F1 (state-of-the-art)
- ACE 2005: ~86.5% F1
- GENIA: ~80.4% F1

**Advantage over Biaffine:**
Consider two overlapping spans with the same boundaries but different inner content:
"Bank of America" (ORG) vs "Bank of the River" (LOC). Biaffine only sees "Bank" and
the last token, while Triaffine additionally considers the middle tokens to make the
correct distinction.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TriaffineNER(NeuralNetworkArchitecture<>,SpanBasedNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Triaffine-NER model in native training mode. |
| `TriaffineNER(NeuralNetworkArchitecture<>,String,SpanBasedNEROptions)` | Creates a Triaffine-NER model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateDefaultLayers` |  |
| `CreateNewInstance` |  |

