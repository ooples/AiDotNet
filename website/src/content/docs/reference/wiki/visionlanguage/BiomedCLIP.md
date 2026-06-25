---
title: "BiomedCLIP<T>"
description: "BiomedCLIP model fine-tuned on 15M biomedical image-text pairs from PubMed Central."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

BiomedCLIP model fine-tuned on 15M biomedical image-text pairs from PubMed Central.

## For Beginners

BiomedCLIP adapts CLIP for the biomedical domain by fine-tuning
on 15 million image-text pairs from PubMed Central articles. It uses ViT-B/16 for images
and PubMedBERT for text, achieving state-of-the-art zero-shot biomedical image classification
for tasks like pathology slide classification and radiology report matching. Default values
follow the original paper settings.

## How It Works

BiomedCLIP (Zhang et al., 2023) adapts CLIP for the biomedical domain using PMC-15M, achieving
state-of-the-art zero-shot biomedical image classification with ViT-B/16 + PubMedBERT.

**References:**

- Paper: "BiomedCLIP: A Multimodal Biomedical Foundation Model" (Zhang et al., 2023)

**Architecture layout:** Mirrors PyTorch / HuggingFace CLIP — vision encoder (patch
embedding + transformer + projection) lives in `Layers`;
text encoder lives in a separate field surfaced through `GetExtraTrainableLayers`.
The default `Predict` + `Tensor{` path is
vision-only — it patch-embeds the input and walks `Layers`. Text-encoder access goes
through `String)`; contrastive image-text similarity through
`String)` / `String[])`.

## Methods

| Method | Summary |
|:-----|:--------|
| `ForwardTextEncoder(Tensor<>)` | Text encoder forward (transformer + projection). |
| `ForwardVisionEncoder(Tensor<>)` | Vision encoder forward (patch embedding + transformer + projection). |
| `GetExtraTrainableLayers` | Surfaces the text-encoder stack to the base weight-registry walker (streaming-offload / weight-pool hooks) without extending the flat parameter APIs (GetParameters / ParameterCount / SetParameters) — those keep the SCOPE CONTRACT (= Layers… |
| `PredictCore(Tensor<>)` | Vision-only forward: walks `Layers` (patch embedding → vision transformer → vision projection) on the preprocessed image. |
| `SyncImageSizeWithArchitecture` | Aligns `_options.ImageSize` with `Architecture.InputHeight` when the architecture declares an explicit square spatial extent. |

