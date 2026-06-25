---
title: "Eagle<T>"
description: "Eagle: NVIDIA data-centric VLM with high-quality training data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Eagle: NVIDIA data-centric VLM with high-quality training data.

## For Beginners

Eagle from NVIDIA focuses on getting the training data right
rather than inventing complex new architectures. It systematically explores which design
choices matter most for vision-language models — including vision encoder selection,
projection strategies, and training data mixtures. The result is a model that achieves
strong performance through careful engineering and high-quality data curation rather than
architectural complexity. It uses a mixture of vision encoders to capture different aspects
of visual information. Default values follow the original paper settings.

## How It Works

Eagle (NVIDIA, 2024) takes a data-centric approach to vision-language model design, systematically
exploring the design space of multimodal LLMs with a focus on high-quality training data curation.
It uses a mixture of vision encoders to capture complementary visual features and demonstrates that
careful data selection and training strategies matter more than architectural novelty for achieving
strong multimodal performance.

**References:**

- Paper: "Eagle: Exploring The Design Space for Multimodal LLMs" (NVIDIA, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Eagle's data-centric multi-encoder fusion. |

