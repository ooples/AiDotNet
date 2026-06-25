---
title: "CatVTONModel<T>"
description: "CatVTON model for concatenation-based virtual try-on without warping modules."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VirtualTryOn`

CatVTON model for concatenation-based virtual try-on without warping modules.

## For Beginners

CatVTON is a simpler approach to virtual try-on. Instead of
complex warping to fit clothes to a body, it simply feeds both the person and garment
images together and lets the AI figure out how to combine them naturally.

## How It Works

CatVTON simplifies virtual try-on by concatenating garment features directly with
person features as input, eliminating the need for explicit warping modules. The diffusion
model learns to implicitly handle deformation and blending.

Reference: Chong et al., "CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models", 2024

