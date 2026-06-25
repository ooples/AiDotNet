---
title: "RADIOv25<T>"
description: "RADIOv2.5 agglomerative vision foundation model distilling DINOv2, SAM, SigLIP, and CLIP."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

RADIOv2.5 agglomerative vision foundation model distilling DINOv2, SAM, SigLIP, and CLIP.

## For Beginners

RADIO v2.5 from NVIDIA is a "best of all worlds" vision encoder
that distills knowledge from multiple teacher models (DINOv2, SAM, SigLIP, and CLIP) into
a single student model. The result is a universal vision backbone that can be used as a
drop-in replacement for any of those individual models, producing compatible features for
all of them. Default values follow the original paper settings.

## How It Works

RADIOv2.5 (Ranzinger et al., NVIDIA 2025) distills multiple teacher vision models into a single
student via multi-teacher distillation. The student produces features compatible with all teachers,
serving as a universal drop-in replacement for DINOv2, SAM, SigLIP, and CLIP vision encoders.

**References:**

- Paper: "AM-RADIO: Agglomerative Vision Foundation Model" (Ranzinger et al., 2025)

