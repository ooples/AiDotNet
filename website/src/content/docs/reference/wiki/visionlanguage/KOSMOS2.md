---
title: "KOSMOS2<T>"
description: "KOSMOS-2: grounded multimodal large language model with text spans linked to bounding boxes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Generative`

KOSMOS-2: grounded multimodal large language model with text spans linked to bounding boxes.

## For Beginners

KOSMOS-2 extends KOSMOS-1 with visual grounding — the ability
to link words in generated text to specific bounding box locations in the image. It uses
special location tokens to encode bounding box coordinates, enabling the model to output
phrases like "the dog <box>x1,y1,x2,y2</box>" that point to objects in the image.
Default values follow the original paper settings.

## How It Works

KOSMOS-2 (Peng et al., 2023) extends KOSMOS-1 with grounding capabilities by linking text spans
to bounding box locations in the image. Special location tokens encode bounding box coordinates,
enabling the model to output referring expressions grounded in the visual input. The architecture
retains the causal multimodal LM design with visual tokens embedded directly in the sequence.

**References:**

- Paper: "Kosmos-2: Grounding Multimodal Large Language Models to the World" (Peng et al., 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using KOSMOS-2's grounded multimodal causal LM architecture. |
| `GetExtraTrainableLayers` |  |

