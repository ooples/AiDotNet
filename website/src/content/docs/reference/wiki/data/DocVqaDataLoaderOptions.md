---
title: "DocVqaDataLoaderOptions"
description: "Configuration options for the DocVQA (Document Visual Question Answering) data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Vision.Benchmarks`

Configuration options for the DocVQA (Document Visual Question Answering) data loader.

## How It Works

DocVQA contains document images with questions and answers. Standard benchmark for
document understanding models combining OCR and visual reasoning.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `ImageHeight` | Image height after resizing. |
| `ImageWidth` | Image width after resizing. |
| `MaxAnswerLength` | Maximum answer character length for text encoding. |
| `MaxQuestionLength` | Maximum question token length. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `Split` | Dataset split to load. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

