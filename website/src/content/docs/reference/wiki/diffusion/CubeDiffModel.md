---
title: "CubeDiffModel<T>"
description: "CubeDiff model for cubemap-based panoramic generation with cross-face consistency."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Panorama`

CubeDiff model for cubemap-based panoramic generation with cross-face consistency.

## For Beginners

CubeDiff creates complete 360-degree environments by generating
six connected views (like the faces of a cube) simultaneously. Each face connects
seamlessly to its neighbors, creating an immersive surround environment perfect for
VR or game backgrounds.

## How It Works

CubeDiff generates cubemap panoramas by jointly denoising all six cube faces with
cross-face attention at the edges. This ensures seamless transitions between faces
and produces high-quality 360x180 degree environments.

