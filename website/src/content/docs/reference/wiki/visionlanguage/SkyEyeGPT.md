---
title: "SkyEyeGPT<T>"
description: "SkyEyeGPT: unified remote sensing vision-language tasks with 968K instruction samples."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.RemoteSensing`

SkyEyeGPT: unified remote sensing vision-language tasks with 968K instruction samples.

## For Beginners

SkyEyeGPT is a vision-language model specialized for satellite
and aerial imagery understanding tasks. Default values follow the original paper settings.

## How It Works

SkyEyeGPT (Zhan et al., 2024) unifies remote sensing vision-language tasks via instruction
tuning with a large language model. Trained on 968K instruction samples covering captioning,
visual question answering, grounding, and scene classification, it adapts a general-purpose
VLM to the remote sensing domain with satellite and aerial imagery understanding.

**References:**

- Paper: "SkyEyeGPT: Unifying Remote Sensing Vision-Language Tasks via Instruction Tuning with Large Language Model (Zhan et al., 2024)"

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from a remote sensing image using SkyEyeGPT's unified multi-task pipeline. |

