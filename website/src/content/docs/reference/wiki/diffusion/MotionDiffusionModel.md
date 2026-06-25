---
title: "MotionDiffusionModel<T>"
description: "Motion Diffusion Model (MDM) for text-to-motion generation of human body movements."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.MotionGeneration`

Motion Diffusion Model (MDM) for text-to-motion generation of human body movements.

## For Beginners

MDM generates 3D human body animations from text. Describe a
motion like "a person doing a jumping jack" and MDM produces the corresponding body
movement animation with realistic joint rotations and timing.

## How It Works

MDM generates human motion sequences from text descriptions using a transformer-based
diffusion model. It operates in joint rotation space, producing temporally coherent
body animations that match the text description.

Reference: Tevet et al., "Human Motion Diffusion Model", ICLR 2023

