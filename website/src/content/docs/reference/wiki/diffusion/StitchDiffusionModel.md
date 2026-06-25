---
title: "StitchDiffusionModel<T>"
description: "StitchDiffusion model for seamless 360-degree panorama generation with wrap-around consistency."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Panorama`

StitchDiffusion model for seamless 360-degree panorama generation with wrap-around consistency.

## For Beginners

StitchDiffusion creates full 360-degree panoramas where the
left edge connects perfectly to the right edge, forming a complete surround view.
Perfect for VR content or immersive environments.

## How It Works

StitchDiffusion generates 360-degree panoramas by ensuring the left and right edges
connect seamlessly. Uses circular padding in the latent space and a stitching loss
to maintain continuity at the wrap-around boundary.

