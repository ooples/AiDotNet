---
title: "IDMVTONModel<T>"
description: "IDM-VTON model for image-based virtual try-on with high-fidelity garment transfer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VirtualTryOn`

IDM-VTON model for image-based virtual try-on with high-fidelity garment transfer.

## For Beginners

IDM-VTON lets you see how clothes would look on a person.
Give it a photo of a person and a photo of a garment, and it generates a realistic
image of the person wearing that garment, preserving both the person's pose and the
garment's details.

## How It Works

IDM-VTON uses a dual U-Net architecture — one encodes the garment image and the other
generates the person wearing the garment. It features IP-Adapter-based garment encoding
and GarmentNet for preserving fine texture details during warping and blending.

Reference: Choi et al., "Improving Diffusion Models for Authentic Virtual Try-on in the Wild", 2024

