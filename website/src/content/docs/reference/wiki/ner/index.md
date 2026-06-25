---
title: "NER"
description: "All 43 public types in the AiDotNet.ner namespace, organized by kind."
section: "API Reference"
---

**43** public types in this namespace, organized by kind.

## Models & Types (33)

| Type | Summary |
|:-----|:--------|
| [`BERTNER<T>`](/docs/reference/wiki/ner/bertner/) | BERT-NER: BERT (Bidirectional Encoder Representations from Transformers) with token classification head for Named Entity Recognition. |
| [`BLINKNER<T>`](/docs/reference/wiki/ner/blinkner/) | BLINK: BERT-based bi-encoder for entity linking and Named Entity Recognition. |
| [`BiLSTMCRF<T>`](/docs/reference/wiki/ner/bilstmcrf/) | BiLSTM-CRF: Bidirectional LSTM with Conditional Random Field for Named Entity Recognition. |
| [`BiaffineNER<T>`](/docs/reference/wiki/ner/biaffinener/) | Biaffine-NER: Named Entity Recognition as dependency parsing using biaffine classifiers. |
| [`BioBERTNER<T>`](/docs/reference/wiki/ner/biobertner/) | BioBERT-NER: Biomedical domain-specific BERT for Named Entity Recognition in biomedical text. |
| [`CNNBiLSTMCRF<T>`](/docs/reference/wiki/ner/cnnbilstmcrf/) | CNN-BiLSTM-CRF: Character CNN + Bidirectional LSTM + Conditional Random Field for Named Entity Recognition. |
| [`ClinicalBERTNER<T>`](/docs/reference/wiki/ner/clinicalbertner/) | ClinicalBERT-NER: Clinical domain BERT for Named Entity Recognition in clinical notes and EHRs. |
| [`DeBERTaNER<T>`](/docs/reference/wiki/ner/debertaner/) | DeBERTa-NER: Decoding-enhanced BERT with disentangled Attention for NER. |
| [`DistilBERTNER<T>`](/docs/reference/wiki/ner/distilbertner/) | DistilBERT-NER: Knowledge-distilled BERT for efficient Named Entity Recognition. |
| [`ELECTRANER<T>`](/docs/reference/wiki/ner/electraner/) | ELECTRA-NER: Efficiently Learning an Encoder that Classifies Token Replacements Accurately for NER. |
| [`FinBERTNER<T>`](/docs/reference/wiki/ner/finbertner/) | FinBERT-NER: Financial domain BERT for Named Entity Recognition in financial text. |
| [`InstructionNER<T>`](/docs/reference/wiki/ner/instructionner/) | InstructionNER: Instruction-tuned transformer for few-shot and zero-shot NER via natural language instructions. |
| [`LSTMCRF<T>`](/docs/reference/wiki/ner/lstmcrf/) | LSTM-CRF: Unidirectional LSTM with Conditional Random Field for Named Entity Recognition. |
| [`LegalBERTNER<T>`](/docs/reference/wiki/ner/legalbertner/) | Legal-BERT-NER: Legal domain BERT for Named Entity Recognition in legal documents. |
| [`NERTrainingProgress`](/docs/reference/wiki/ner/nertrainingprogress/) | Reports training progress for NER models, including loss, F1 score, and epoch information. |
| [`NerTextEncoder`](/docs/reference/wiki/ner/nertextencoder/) | Converts raw text into the packed integer indices consumed by `WordCharEmbeddingLayer` / `WordCharBiLSTMCRF`, and builds the word and character vocabularies. |
| [`ONNXNER<T>`](/docs/reference/wiki/ner/onnxner/) | ONNX-NER: Generic ONNX Runtime-based NER model for high-performance inference with any exported NER model. |
| [`PURENER<T>`](/docs/reference/wiki/ner/purener/) | PURE: Princeton University Relation Extraction - pipeline approach for joint entity and relation extraction. |
| [`PromptNER<T>`](/docs/reference/wiki/ner/promptner/) | PromptNER: Prompt-based learning for few-shot Named Entity Recognition. |
| [`PubMedBERTNER<T>`](/docs/reference/wiki/ner/pubmedbertner/) | PubMedBERT-NER: PubMed domain-specific BERT pre-trained from scratch on biomedical text for NER. |
| [`PyramidNER<T>`](/docs/reference/wiki/ner/pyramidner/) | Pyramid-NER: Hierarchical pyramid network for nested Named Entity Recognition. |
| [`RELNER<T>`](/docs/reference/wiki/ner/relner/) | REL: Radboud Entity Linker - end-to-end entity linking combining NER with entity disambiguation. |
| [`RoBERTaNER<T>`](/docs/reference/wiki/ner/robertaner/) | RoBERTa-NER: Robustly Optimized BERT Approach with token classification for NER. |
| [`SECBertNER<T>`](/docs/reference/wiki/ner/secbertner/) | SEC-BERT-NER: Securities and Exchange Commission domain BERT for NER in regulatory filings. |
| [`SciBERTNER<T>`](/docs/reference/wiki/ner/scibertner/) | SciBERT-NER: Scientific domain BERT for Named Entity Recognition in scientific literature. |
| [`SpERTNER<T>`](/docs/reference/wiki/ner/spertner/) | SpERT: Span-based Entity and Relation Transformer for joint entity and relation extraction. |
| [`SpanBERTNER<T>`](/docs/reference/wiki/ner/spanbertner/) | SpanBERT-NER: Span-level BERT pre-training with token classification for NER. |
| [`TemplateNER<T>`](/docs/reference/wiki/ner/templatener/) | Template-NER: Template-based prompt approach for few-shot and zero-shot NER. |
| [`TinyBERTNER<T>`](/docs/reference/wiki/ner/tinybertner/) | TinyBERT-NER: Two-stage distilled BERT for ultra-efficient Named Entity Recognition. |
| [`TriaffineNER<T>`](/docs/reference/wiki/ner/triaffinener/) | Triaffine-NER: Three-way interaction model for nested Named Entity Recognition. |
| [`W2NER<T>`](/docs/reference/wiki/ner/w2ner/) | W2NER: Word-Word Relation Classification for unified flat and nested NER. |
| [`WordCharBiLSTMCRF<T>`](/docs/reference/wiki/ner/wordcharbilstmcrf/) | Paper-faithful word + character BiLSTM-CRF for Named Entity Recognition (Lample et al., NAACL 2016, "Neural Architectures for Named Entity Recognition"). |
| [`XLMRoBERTaNER<T>`](/docs/reference/wiki/ner/xlmrobertaner/) | XLM-RoBERTa-NER: Cross-lingual RoBERTa for multilingual Named Entity Recognition. |

## Base Classes (4)

| Type | Summary |
|:-----|:--------|
| [`NERNeuralNetworkBase<T>`](/docs/reference/wiki/ner/nerneuralnetworkbase/) | Base class for NER-focused neural networks that can operate in both ONNX inference and native training modes. |
| [`SequenceLabelingNERBase<T>`](/docs/reference/wiki/ner/sequencelabelingnerbase/) | Base class for sequence labeling NER models that assign a BIO label to each token in a sequence. |
| [`SpanBasedNERBase<T>`](/docs/reference/wiki/ner/spanbasednerbase/) | Base class for span-based NER models (SpERT, BiaffineNER, PURE). |
| [`TransformerNERBase<T>`](/docs/reference/wiki/ner/transformernerbase/) | Base class for transformer-based NER models (BERT-NER, RoBERTa-NER, DeBERTa-NER, etc.). |

## Interfaces (1)

| Type | Summary |
|:-----|:--------|
| [`INERModel<T>`](/docs/reference/wiki/ner/inermodel/) | Base interface for all Named Entity Recognition (NER) AI models in AiDotNet. |

## Options & Configuration (5)

| Type | Summary |
|:-----|:--------|
| [`BiLSTMCRFOptions`](/docs/reference/wiki/ner/bilstmcrfoptions/) | Configuration options for the BiLSTM-CRF Named Entity Recognition model. |
| [`CNNBiLSTMCRFOptions`](/docs/reference/wiki/ner/cnnbilstmcrfoptions/) | Configuration options for the CNN-BiLSTM-CRF Named Entity Recognition model. |
| [`LSTMCRFOptions`](/docs/reference/wiki/ner/lstmcrfoptions/) | Configuration options for the LSTM-CRF Named Entity Recognition model. |
| [`SpanBasedNEROptions`](/docs/reference/wiki/ner/spanbasedneroptions/) | Options shared by span-based NER models (SpERT, BiaffineNER, PURE). |
| [`TransformerNEROptions`](/docs/reference/wiki/ner/transformerneroptions/) | Base configuration options shared by all transformer-based NER models (BERT-NER, RoBERTa-NER, DeBERTa-NER, ELECTRA-NER, etc.). |

