---
title: "MedicalASR<T>"
description: "Medical ASR: domain-specialized medical speech recognition"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Specialized`

Medical ASR: domain-specialized medical speech recognition

## For Beginners

Medical ASR is specialized for clinical dictation, pathology reports, and medical conversations. The model is fine-tuned on medical speech corpora covering diverse medical specialties, drug names, procedures, and diagnostic terminology. A medical ...

## How It Works

**References:**

- Paper: "Domain-Specific Speech Recognition for Medical Dictation" (2024)

Medical ASR is specialized for clinical dictation, pathology reports, and medical conversations. The model is fine-tuned on medical speech corpora covering diverse medical specialties, drug names, procedures, and diagnostic terminology. A medical language model provides domain-specific rescoring to handle complex medical vocabulary. The system supports real-time clinical documentation with HIPAA-compliant processing and achieves significantly lower WER on medical speech than general-purpose ASR models.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes medical speech using domain-specialized Conformer encoder. |

