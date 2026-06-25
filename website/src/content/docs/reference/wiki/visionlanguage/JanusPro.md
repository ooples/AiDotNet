---
title: "JanusPro<T>"
description: "Janus-Pro: unified multimodal understanding and generation with decoupled vision encoders (Chen et al., DeepSeek 2025, arXiv:2501.17811)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Unified`

Janus-Pro: unified multimodal understanding and generation with decoupled vision encoders
(Chen et al., DeepSeek 2025, arXiv:2501.17811).

## For Beginners

Janus-Pro is the first model that does BOTH "understand image, answer
in text" AND "describe in text, generate image" with one unified backbone — but it uses two
completely different vision encoders for each direction, which the paper shows is much better than
trying to share. Default values follow the published 1.5B configuration (scale up via
`JanusProOptions` for the 7B variant).

## How It Works

Janus-Pro is the scaled-up successor to Janus (Wu et al. 2024, arXiv:2410.13848). Both
models share Janus's central design insight: **vision encoding for understanding**
(an image-to-language path that feeds SigLIP-style continuous features into the LLM) and
**vision encoding for generation** (a VQ-VAE codebook that turns the LLM's output
token stream back into pixels) are *fully decoupled*. The two paths converge only at
the autoregressive transformer backbone in the middle. Janus-Pro adds: a 16384-entry VQ
codebook (vs Janus's 8192), curriculum-based training, expanded synthetic data, and a
7B-parameter DeepSeek-LLM backbone.

**Paper-faithful pieces implemented here:**

**What is NOT verified in-session:**

## Properties

| Property | Summary |
|:-----|:--------|
| `CfgScale` | Classifier-free guidance scale used during autoregressive image generation. |
| `GenerationTokenCount` | Number of generation-side VQ tokens in the output grid (24×24 for 384px output, matching paper §3.3). |
| `ParameterCount` |  |
| `VQCodebook` | VQ codebook used by the generation path. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetokenizeVQTokens(Int32[])` | VQ-VAE detokenizer: token grid → codebook embeddings → deconv upsampling stack → pixels. |
| `EmbedPromptTokens(Tensor<>)` | Looks up prompt-token embeddings through the learned `_tokenEmbedding` table (Chen et al. |
| `EncodeImage(Tensor<>)` | Janus-Pro **understanding** path: image → SigLIP-style continuous features → LLM hidden state. |
| `GenerateFromImage(Tensor<>,String)` | Image-to-text (understanding) forward pass. |
| `GenerateImage(String)` | Text-to-image (generation) forward pass. |
| `GenerationModules` | The learned generation modules in their FIXED flat-parameter/serialization order. |
| `GetParameters` |  |
| `ProjectCodebookEmbeddingToDecoderDim(Tensor<>)` | Projects a VQ codebook embedding (dimension `EmbeddingDim`) up to the LLM decoder dimension through the learned `_codebookProjection` dense layer. |
| `SetParameters(Vector<>)` |  |

