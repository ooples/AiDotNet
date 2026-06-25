---
title: "RBModulationModel<T>"
description: "RB-Modulation model for training-free style transfer via reference-based attention modulation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.StyleTransfer`

RB-Modulation model for training-free style transfer via reference-based attention modulation.

## For Beginners

RB-Modulation is a "plug and play" style transfer — no training
needed. Just provide a style image and a content image, and it transfers the style by
cleverly manipulating the attention patterns inside the model. Works with any pre-trained
diffusion model.

## How It Works

RB-Modulation enables style transfer without any training or fine-tuning by modulating
self-attention layers using features extracted from the style reference image. Style
features replace content features at specific attention layers during generation.

Reference: Rout et al., "RB-Modulation: Training-Free Personalization of Diffusion Models using Stochastic Optimal Control", 2024

