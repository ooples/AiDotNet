---
title: "GroundedAnswer<T>"
description: "Represents a generated answer with citations and source attribution for transparency and verification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Models`

Represents a generated answer with citations and source attribution for transparency and verification.

## For Beginners

A GroundedAnswer is like a research paper answer with footnotes.

Think of it like writing a school report:

- Answer: Your written response to the question
- SourceDocuments: The books and articles you referenced
- Citations: The footnote numbers [1], [2], [3] in your text
- ConfidenceScore: How sure you are that your answer is correct

Why is this important?

- Transparency: You can see where the information came from
- Verification: You can check if the AI interpreted sources correctly
- Trust: Answers backed by real sources are more reliable
- Learning: You can read the sources to learn more

Example:
Question: "What causes rainbows?"
Answer: "Rainbows are caused by light refraction through water droplets [1]. 
The water acts like a prism, separating white light into colors [2]."
Citations: [1] = Physics textbook, [2] = Optics journal article
SourceDocuments: The actual textbook and article
ConfidenceScore: 0.95 (95% confident this answer is accurate)

## How It Works

A GroundedAnswer contains the AI-generated response along with references to the source documents
used to create it. This grounding enables users to verify claims, understand context, and trust
the generated content. The answer includes both inline citations and references to source documents.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GroundedAnswer` | Initializes a new instance of the GroundedAnswer class. |
| `GroundedAnswer(String,IReadOnlyList<Document<>>)` | Initializes a new instance of the GroundedAnswer class with basic components. |
| `GroundedAnswer(String,String,IReadOnlyList<Document<>>,IReadOnlyList<String>,Double)` | Initializes a new instance of the GroundedAnswer class with all components. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Answer` | Gets or sets the generated answer text. |
| `Citations` | Gets or sets the extracted citations mapping citation markers to source documents. |
| `ConfidenceScore` | Gets or sets the confidence score indicating the model's certainty in the answer. |
| `Query` | Gets or sets the original query that prompted this answer. |
| `SourceDocuments` | Gets or sets the source documents that were used to generate the answer. |

