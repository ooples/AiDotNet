---
title: "PatchEmbedHelper"
description: "ViT-style patch embedding (Dosovitskiy et al."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

ViT-style patch embedding (Dosovitskiy et al. 2020 §3.1): turns raw NCHW
image input into a BSC token sequence for transformer-backed vision
encoders (PaLM-E, BiomedCLIP, PaLI-3, OpenCLIP, ViT). Already-tokenized
inputs pass through unchanged.

