---
title: "IMedicalVLM<T>"
description: "Interface for medical domain vision-language models specializing in biomedical image understanding."
section: "API Reference"
---

`Interfaces` · `AiDotNet.VisionLanguage.Interfaces`

Interface for medical domain vision-language models specializing in biomedical image understanding.

## How It Works

Medical VLMs are trained on biomedical image-text pairs for tasks such as radiology report generation,
medical visual question answering, pathology analysis, and clinical decision support.

## Properties

| Property | Summary |
|:-----|:--------|
| `LanguageModelName` | Gets the name of the language model backbone. |
| `MedicalDomain` | Gets the medical domain this model specializes in (e.g., "Radiology", "Pathology", "General"). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerMedicalQuestion(Tensor<>,String)` | Answers a medical question about a biomedical image. |

