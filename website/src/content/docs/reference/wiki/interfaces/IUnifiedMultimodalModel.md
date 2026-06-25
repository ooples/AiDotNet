---
title: "IUnifiedMultimodalModel<T>"
description: "Defines the contract for unified multimodal models that handle multiple modalities in a single architecture, similar to GPT-4o, Gemini, or Meta's CM3Leon."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for unified multimodal models that handle multiple modalities
in a single architecture, similar to GPT-4o, Gemini, or Meta's CM3Leon.

## For Beginners

One model that can see, hear, read, and create!

Key capabilities:

- Any-to-any generation: Text → Image, Image → Text, Audio → Text, etc.
- Interleaved understanding: Process mixed sequences of text, images, audio
- Cross-modal reasoning: Answer questions using information from multiple sources
- Unified embeddings: All modalities share a common representation space

Architecture concepts:

1. Modality Encoders: Specialized encoders for each input type
2. Unified Transformer: Core model that processes all modalities
3. Modality Decoders: Generate outputs in any modality
4. Cross-Attention: Allow modalities to attend to each other

## How It Works

Unified multimodal models represent the next generation of AI systems that can
seamlessly process and generate content across multiple modalities (text, image,
audio, video) within a single unified architecture.

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the unified embedding dimension. |
| `MaxSequenceLength` | Gets the maximum sequence length for interleaved inputs. |
| `SupportedInputModalities` | Gets the supported input modalities. |
| `SupportedOutputModalities` | Gets the supported output modalities. |
| `SupportsStreaming` | Gets whether the model supports streaming generation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AlignTemporally(IEnumerable<MultimodalInput<>>)` | Aligns content across modalities temporally. |
| `AnswerQuestion(IEnumerable<MultimodalInput<>>,String)` | Answers a question using multimodal context. |
| `Chat(IEnumerable<ValueTuple<String,IEnumerable<MultimodalInput<>>>>,IEnumerable<MultimodalInput<>>,Int32)` | Conducts a multi-turn conversation with multimodal inputs. |
| `Compare(IEnumerable<MultimodalInput<>>,IEnumerable<String>)` | Compares multiple multimodal inputs and provides analysis. |
| `ComputeSimilarity(MultimodalInput<>,MultimodalInput<>)` | Computes cross-modal similarity between inputs. |
| `Detect(IEnumerable<MultimodalInput<>>,String)` | Detects and localizes objects/events across modalities. |
| `Edit(MultimodalInput<>,String)` | Edits multimodal content based on instructions. |
| `Encode(MultimodalInput<>)` | Encodes any supported modality into the unified embedding space. |
| `EncodeSequence(IEnumerable<MultimodalInput<>>)` | Encodes multiple interleaved inputs into a sequence of embeddings. |
| `FewShotLearn(IEnumerable<ValueTuple<IEnumerable<MultimodalInput<>>,MultimodalOutput<>>>,IEnumerable<MultimodalInput<>>)` | Performs in-context learning from multimodal examples. |
| `Fuse(IEnumerable<MultimodalInput<>>,String)` | Fuses multiple modality inputs into a unified representation. |
| `Generate(IEnumerable<MultimodalInput<>>,ModalityType,Int32)` | Generates output in the specified modality given multimodal inputs. |
| `GenerateAudio(IEnumerable<MultimodalInput<>>,Double,Int32)` | Generates audio from multimodal inputs. |
| `GenerateImage(IEnumerable<MultimodalInput<>>,Int32,Int32)` | Generates an image from multimodal inputs. |
| `GenerateInterleaved(IEnumerable<MultimodalInput<>>,IEnumerable<ValueTuple<ModalityType,Int32>>)` | Generates an interleaved sequence of multiple modalities. |
| `GenerateText(IEnumerable<MultimodalInput<>>,String,Int32,Double)` | Generates text response from multimodal inputs. |
| `GetCrossModalAttention(IEnumerable<MultimodalInput<>>)` | Gets attention weights showing cross-modal relationships. |
| `Reason(IEnumerable<MultimodalInput<>>,String)` | Performs reasoning across multiple modalities. |
| `Retrieve(MultimodalInput<>,IEnumerable<MultimodalInput<>>,Int32)` | Retrieves the most similar items from a database given a query. |
| `SafetyCheck(IEnumerable<MultimodalInput<>>)` | Checks content for safety across all modalities. |
| `Summarize(IEnumerable<MultimodalInput<>>,ModalityType,Int32)` | Summarizes multimodal content. |
| `Translate(MultimodalInput<>,ModalityType)` | Translates content between modalities. |

