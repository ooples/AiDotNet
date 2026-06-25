---
title: "Postprocessing"
description: "All 18 public types in the AiDotNet.postprocessing namespace, organized by kind."
section: "API Reference"
---

**18** public types in this namespace, organized by kind.

## Models & Types (13)

| Type | Summary |
|:-----|:--------|
| [`DocumentSchema`](/docs/reference/wiki/postprocessing/documentschema/) | Schema for document validation. |
| [`Entity`](/docs/reference/wiki/postprocessing/entity/) | Represents an extracted entity from document text. |
| [`EntityLinking<T>`](/docs/reference/wiki/postprocessing/entitylinking/) | EntityLinking - Entity extraction and linking for document text. |
| [`FieldExtractionRule`](/docs/reference/wiki/postprocessing/fieldextractionrule/) | Rule for extracting a field from document text. |
| [`InvoiceData`](/docs/reference/wiki/postprocessing/invoicedata/) | Structured invoice data extracted from documents. |
| [`InvoiceLineItem`](/docs/reference/wiki/postprocessing/invoicelineitem/) | Invoice line item. |
| [`PostprocessingPipeline<T, TInput, TOutput>`](/docs/reference/wiki/postprocessing/postprocessingpipeline/) | Chains multiple data transformers into a sequential postprocessing pipeline. |
| [`ReceiptData`](/docs/reference/wiki/postprocessing/receiptdata/) | Structured receipt data extracted from documents. |
| [`ReceiptItem`](/docs/reference/wiki/postprocessing/receiptitem/) | Receipt line item. |
| [`SpellCorrection<T>`](/docs/reference/wiki/postprocessing/spellcorrection/) | SpellCorrection - Spell checking and correction for OCR output. |
| [`StructuredOutputParser<T>`](/docs/reference/wiki/postprocessing/structuredoutputparser/) | StructuredOutputParser - Parses document AI outputs into structured data formats. |
| [`TextPostprocessor<T>`](/docs/reference/wiki/postprocessing/textpostprocessor/) | TextPostprocessor - OCR text postprocessing utilities. |
| [`ValidationResult`](/docs/reference/wiki/postprocessing/validationresult/) | Result of document schema validation. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`PostprocessorBase<T, TInput, TOutput>`](/docs/reference/wiki/postprocessing/postprocessorbase/) | Abstract base class for all postprocessors providing common functionality. |

## Enums (2)

| Type | Summary |
|:-----|:--------|
| [`DataType`](/docs/reference/wiki/postprocessing/datatype/) | Data types for parsed document values. |
| [`EntityType`](/docs/reference/wiki/postprocessing/entitytype/) | Types of entities that can be extracted from documents. |

## Options & Configuration (1)

| Type | Summary |
|:-----|:--------|
| [`TextPostprocessorOptions`](/docs/reference/wiki/postprocessing/textpostprocessoroptions/) | Options for text postprocessing. |

## Helpers & Utilities (1)

| Type | Summary |
|:-----|:--------|
| [`PostprocessingRegistry<T, TOutput>`](/docs/reference/wiki/postprocessing/postprocessingregistry/) | Global registry for the postprocessing pipeline. |

