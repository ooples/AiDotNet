---
title: "RT2<T>"
description: "RT-2: vision-language-action model that transfers web knowledge to robotic control (Brohan et al., Google DeepMind, 2023, arXiv:2307.15818)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Robotics`

RT-2: vision-language-action model that transfers web knowledge to robotic control
(Brohan et al., Google DeepMind, 2023, arXiv:2307.15818).

## For Beginners

RT-2 is the model that proved a generic Internet-trained vision-language
model can drive a real robot just by treating "move arm forward 3cm" as a sentence to generate,
rather than by training a separate policy network. Default values follow the published 55B-parameter
PaLI-X configuration but can be scaled down via `RT2Options`.

## How It Works

RT-2 fine-tunes a large vision-language model (PaLI-X or PaLM-E) on robot demonstration data
by representing continuous robot actions as text tokens. Each action dimension is uniformly
discretized into 256 bins (paper §3.2) and each bin is mapped to one of the 256 least
frequently used vocabulary tokens. The VLM then emits these tokens autoregressively just as
it would emit ordinary text, so all of the model's web-scale knowledge transfers directly to
robot control. This class composes a generic ViT-style vision encoder, MLP projection, LLM
decoder, action-bin head and full-vocabulary projection so the same forward path can serve
both vision-language reasoning and robotic action generation, mirroring the encoder-decoder
PaLI/PaLM-E backbone used in the paper.

**Paper-faithful pieces implemented here:**

**What is NOT verified in-session:**

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionTokenizer` | Action tokenizer used during decode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AppendActionTokenEmbedding(Tensor<>,Int32)` | Appends the learned embedding of the most-recently-generated action-bin token to the running decoder context. |
| `EmbedTokenIds(Tensor<>)` | Looks up token embeddings through the shared trainable `_tokenEmbedding` layer. |
| `FuseVisualAndTextEmbeddings(Tensor<>,Tensor<>)` | Concatenates visual encoder output with instruction-token embeddings along the sequence dimension to form the PaLI-style joint context vector per Brohan et al. |
| `GenerateFromImage(Tensor<>,String)` | Runs the full encoder + decoder pipeline once and returns vocabulary-sized logits at the final position. |
| `PredictAction(Tensor<>,String)` | Predicts a continuous robot action using RT-2's action-as-text formulation (Brohan et al. |

