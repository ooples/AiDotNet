---
title: "VILA<T>"
description: "VILA: VLM pre-trained with interleaved image-text data for enhanced in-context learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

VILA: VLM pre-trained with interleaved image-text data for enhanced in-context learning.

## For Beginners

VILA from NVIDIA explores what makes pre-training work best
for vision-language models. Its key finding is that training with interleaved image-text
data — where images and text alternate naturally like on a web page — gives the model
much better in-context learning abilities. This means you can show VILA a few examples
of a new task in the prompt (like "here are 3 examples of product descriptions from
product photos") and it will learn to do that task without any fine-tuning. It also
found that keeping the language model unfrozen during pre-training is essential for
this capability. Default values follow the original paper settings.

## How It Works

VILA (NVIDIA, 2024) investigates pre-training strategies for vision-language models and
demonstrates that pre-training with interleaved image-text data (where images and text
alternate naturally, like web pages) significantly improves in-context learning capabilities.
The model shows that unfreezing the language model during pre-training and using interleaved
data together produce strong few-shot learning performance, where the model can learn new
tasks from just a few examples shown in the prompt.

**References:**

- Paper: "VILA: On Pre-training for Visual Language Models" (NVIDIA, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using VILA's interleaved pre-training with visual instruction tuning. |

