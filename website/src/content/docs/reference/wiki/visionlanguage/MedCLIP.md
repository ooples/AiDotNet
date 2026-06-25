---
title: "MedCLIP<T>"
description: "MedCLIP model using decoupled semantic matching for medical image-text alignment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

MedCLIP model using decoupled semantic matching for medical image-text alignment.

## For Beginners

MedCLIP adapts CLIP for medical imaging by solving a key
problem: medical datasets are small and images are not always paired with matching text.
It uses decoupled semantic matching — any medical image can be paired with any text that
shares the same medical concepts (like "pneumonia" or "chest X-ray"), allowing it to learn
from unpaired data. Default values follow the original paper settings.

## How It Works

MedCLIP (Wang et al., 2022) addresses limited medical data by decoupling image-text inputs:
any image can be paired with any text sharing the same medical concepts (diagnosis, anatomy),
using a semantic matching loss alongside contrastive learning.

**References:**

- Paper: "MedCLIP: Contrastive Learning from Unpaired Medical Images and Text" (Wang et al., EMNLP 2022)

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExtraTrainableLayers` |  |

