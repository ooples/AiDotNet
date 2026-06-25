---
title: "LegalBERTNER<T>"
description: "Legal-BERT-NER: Legal domain BERT for Named Entity Recognition in legal documents."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

Legal-BERT-NER: Legal domain BERT for Named Entity Recognition in legal documents.

## For Beginners

Legal-BERT is BERT trained on court opinions, contracts, and legislation.
It understands legal jargon, citation formats, and legal entity types that general BERT
struggles with. Use Legal-BERT-NER for processing contracts, court filings, regulations,
or any legal documents where entity extraction is needed.

## How It Works

Legal-BERT-NER (Chalkidis et al., EMNLP 2020 Findings - "LEGAL-BERT: The Muppets straight
out of Law School") is BERT pre-trained on 12GB of diverse English legal text for
domain-specific legal NLP tasks including NER.

**Pre-training Data:**

- EU legislation (EU Acquis, treaties, regulations, directives)
- US court opinions (case law from all federal courts)
- US contracts (EDGAR-sourced commercial contracts)
- Legal academic papers (selected law review articles)
- ~12GB total, ~2.5B tokens of legal text

**Legal NER Entity Types:**

- **Court:** Supreme Court, District Court, European Court of Justice
- **Judge:** Justice Roberts, Judge Smith
- **Legislation:** Article 5(1), Section 230, Title VII
- **Citation:** Brown v. Board of Education, 347 U.S. 483 (1954)
- **Party:** Plaintiff, defendant, appellant names
- **Legal Concept:** Due process, habeas corpus, res judicata
- **Jurisdiction:** State of New York, European Union, federal

**Why Legal NER Needs Domain Models:**
Legal text has unique structure: nested citations ("see also 42 U.S.C. 1983"), Latin terms
(habeas corpus, stare decisis), and specific entity patterns (case citations like
"548 F.3d 290 (2d Cir. 2008)"). Legal-BERT understands these patterns.

**Performance:**

- Legal NER: ~88-91% F1 (vs general BERT ~82-85% on legal text)
- Contract entity extraction: ~86-89% F1

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LegalBERTNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates a Legal-BERT-NER model in ONNX inference mode. |
| `LegalBERTNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Legal-BERT-NER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

