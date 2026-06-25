---
title: "MambaModelState<T>"
description: "Per-token (KV-cached) decoding state for a `MambaLanguageModel`: one `MambaStepState` per Mamba block, in layer order."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Per-token (KV-cached) decoding state for a `MambaLanguageModel`: one
`MambaStepState` per Mamba block, in layer order. All other layers (embedding, norm, LM
head) are position-wise and need no carried state.

