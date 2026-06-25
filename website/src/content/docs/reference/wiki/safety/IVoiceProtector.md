---
title: "IVoiceProtector<T>"
description: "Interface for voice protection modules that defend against voice cloning and deepfake attacks."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Safety.Audio`

Interface for voice protection modules that defend against voice cloning and deepfake attacks.

## For Beginners

A voice protector adds invisible protection to voice recordings
so that AI voice cloning tools cannot accurately copy the voice. The protection is
designed to be inaudible to humans but disruptive to cloning algorithms.

## How It Works

Voice protectors apply active defense techniques to audio to prevent unauthorized
voice cloning. Approaches include adding imperceptible perturbations (SPEC),
embedding watermarks (AudioSeal), and psychoacoustic masking (VocalCrypt).

**References:**

- SafeSpeech: SPEC perturbation framework against voice cloning (2025, arxiv:2504.09839)
- VocalCrypt: Pseudo-timbre jamming for voice protection (2025, arxiv:2502.10329)
- AudioSeal: Localized watermarking (Meta AI, 2024, arxiv:2401.17264)

## Methods

| Method | Summary |
|:-----|:--------|
| `ProtectVoice(Vector<>,Int32)` | Applies voice protection to the given audio samples. |

