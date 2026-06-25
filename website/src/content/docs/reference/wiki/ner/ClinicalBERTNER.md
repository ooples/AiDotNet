---
title: "ClinicalBERTNER<T>"
description: "ClinicalBERT-NER: Clinical domain BERT for Named Entity Recognition in clinical notes and EHRs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

ClinicalBERT-NER: Clinical domain BERT for Named Entity Recognition in clinical notes and EHRs.

## For Beginners

ClinicalBERT is BERT trained on clinical notes from hospitals. It
understands medical abbreviations, drug names, diagnoses, and the unique writing style of
doctors and nurses. Use ClinicalBERT-NER for extracting medical entities from clinical notes,
discharge summaries, or electronic health records. Note: HIPAA compliance and data privacy
are critical when processing real clinical data.

## How It Works

ClinicalBERT-NER (Alsentzer et al., NAACL 2019 Clinical NLP Workshop - "Publicly Available
Clinical BERT Embeddings"; Huang et al., 2019 - "ClinicalBERT: Modeling Clinical Notes and
Predicting Hospital Readmission") is BERT further pre-trained on clinical text from
electronic health records (EHRs) for clinical NLP tasks.

**Pre-training Data:**

- MIMIC-III clinical notes (~880M words from 2M+ clinical notes)
- Discharge summaries, radiology reports, nursing notes
- Physician progress notes, operative reports
- Initialized from BioBERT weights (which were initialized from BERT)

**Clinical NER Entity Types:**

- **Problem/Diagnosis:** Type 2 diabetes mellitus, acute myocardial infarction
- **Treatment:** Metformin 500mg, coronary artery bypass graft
- **Test:** Complete blood count, chest X-ray, echocardiogram
- **Anatomy:** Left ventricle, right lower lobe, anterior cruciate ligament
- **Dosage:** 500mg BID, 10mg/kg/day
- **Duration:** 7-day course, q6h for 48 hours
- **Temporal:** Post-operative day 3, on admission, at discharge

**Why Clinical NER is Different:**
Clinical text is uniquely challenging: heavy abbreviations (SOB = shortness of breath, not
an expletive), misspellings, fragmented sentences, negation (patient denies chest pain),
and domain-specific jargon (PRN, NPO, PERRLA). ClinicalBERT handles these patterns.

**Performance:**

- i2b2 2010 Clinical NER: ~88.5% F1 (vs BERT ~84.3%, BioBERT ~86.1%)
- i2b2 2012 Temporal NER: ~85.2% F1
- n2c2 2018 Medication NER: ~91.3% F1

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClinicalBERTNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates a ClinicalBERT-NER model in ONNX inference mode. |
| `ClinicalBERTNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a ClinicalBERT-NER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

