---
title: "StableVITONModel<T>"
description: "StableVITON model for learning semantic correspondence with Stable Diffusion for virtual try-on."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VirtualTryOn`

StableVITON model for learning semantic correspondence with Stable Diffusion for virtual try-on.

## For Beginners

StableVITON uses Stable Diffusion (a powerful image generator)
to handle virtual try-on. It learns how garment parts correspond to body parts,
producing very realistic results because it builds on SD's strong image generation.

## How It Works

StableVITON leverages pretrained Stable Diffusion with zero cross-attention blocks
to learn spatial correspondence between garment and person images. Fine-tunes with
ControlNet-like conditioning for realistic garment transfer.

Reference: Kim et al., "StableVITON: Learning Semantic Correspondence with Latent Diffusion Model for Virtual Try-On", CVPR 2024

