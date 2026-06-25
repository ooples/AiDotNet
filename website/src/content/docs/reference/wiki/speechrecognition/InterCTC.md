---
title: "InterCTC<T>"
description: "Inter-CTC: intermediate CTC loss for deep encoder training"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.CTCVariants`

Inter-CTC: intermediate CTC loss for deep encoder training

## For Beginners

InterCTC applies auxiliary CTC losses at intermediate encoder layers, not just the final layer. Each intermediate CTC head provides gradient signals to lower layers, improving training of deep encoders. The intermediate predictions can also be use...

## How It Works

**References:**

- Paper: "Intermediate Loss Regularization for CTC-based Speech Recognition" (Lee and Watanabe, 2021)

InterCTC applies auxiliary CTC losses at intermediate encoder layers, not just the final layer. Each intermediate CTC head provides gradient signals to lower layers, improving training of deep encoders. The intermediate predictions can also be used for self-conditioning: each layer receives the CTC posteriors from the layer below, enabling iterative refinement of predictions through the encoder stack. This consistently improves accuracy for deep Conformer encoders (12+ layers).

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Conformer with intermediate CTC losses. |

