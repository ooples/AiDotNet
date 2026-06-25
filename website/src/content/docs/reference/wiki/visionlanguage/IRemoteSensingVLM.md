---
title: "IRemoteSensingVLM<T>"
description: "Interface for remote sensing vision-language models specializing in satellite and aerial imagery."
section: "API Reference"
---

`Interfaces` · `AiDotNet.VisionLanguage.Interfaces`

Interface for remote sensing vision-language models specializing in satellite and aerial imagery.

## How It Works

Remote sensing VLMs understand satellite imagery, aerial photos, and geospatial data.
They support tasks like scene classification, object detection in overhead imagery,
change detection, and grounded visual question answering on geospatial data.

## Properties

| Property | Summary |
|:-----|:--------|
| `LanguageModelName` | Gets the name of the language model backbone. |
| `SupportedBands` | Gets the supported image resolution bands (e.g., RGB, multispectral). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerRemoteSensingQuestion(Tensor<>,String)` | Answers a question about a remote sensing image. |

