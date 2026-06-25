---
title: "BLIP3<T>"
description: "BLIP-3 (xGen-MM): scaled vision-language model with interleaved data and any-to-any generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Generative`

BLIP-3 (xGen-MM): scaled vision-language model with interleaved data and any-to-any generation.

## For Beginners

BLIP-3 (also called xGen-MM) is Salesforce's scaled multimodal model
that can understand images, answer questions about them, and generate both text and images from
interleaved image-text inputs. It builds on BLIP-2's Q-Former design with larger capacity and
training on web-scraped interleaved data. Default values follow the original paper settings.

## How It Works

BLIP-3/xGen-MM (Salesforce, 2024) scales the BLIP-2 architecture with interleaved image-text
data (OBELICS), larger Q-Former capacity, and any-to-any generation capabilities.

**References:**

- Paper: "xGen-MM (BLIP-3): A Family of Open Large Multimodal Models" (Salesforce, 2024)

**Architecture layout:** Triple-stream — vision encoder lives in
`Layers` (so default Predict / TrainWithTape walk only it),
Q-Former lives in a private auxiliary stream, decoder lives in another. Image+text generation
goes through `String)` which walks all three streams; raw
`Predict` returns the vision-only forward (matching the IVisualEncoder contract).

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using BLIP-3/xGen-MM's scaled Q-Former architecture. |
| `GetExtraTrainableLayers` |  |

