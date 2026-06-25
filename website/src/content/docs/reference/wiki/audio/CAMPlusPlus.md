---
title: "CAMPlusPlus<T>"
description: "CAM++ (Context-Aware Masking Plus Plus) speaker verification model (Wang et al., 2023)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Speaker`

CAM++ (Context-Aware Masking Plus Plus) speaker verification model (Wang et al., 2023).

## For Beginners

CAM++ is like a smart listener who can automatically tune out
background noise and silence, focusing only on the parts of audio where someone is actually
speaking. It creates a unique "voiceprint" from just the speech portions, making it both
fast and accurate for identifying people by their voice.

**Usage:**

## How It Works

CAM++ is a fast speaker verification model using context-aware masking with a densely
connected TDNN (D-TDNN). It learns to mask uninformative frames (silence, noise) and
focus on speech-rich segments. Achieves competitive EER while being significantly faster
than Transformer-based approaches.

