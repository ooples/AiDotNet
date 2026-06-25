---
title: "NemotronSpeech<T>"
description: "Nemotron-Speech: NVIDIA's multi-task ASR model with Nemotron LLM backbone."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.NeMo`

Nemotron-Speech: NVIDIA's multi-task ASR model with Nemotron LLM backbone.

## For Beginners

Nemotron-Speech uses a Fast Conformer encoder paired with a Nemotron LLM decoder for multi-task speech understanding: ASR, translation, summarization, and instruction-following on audio input. The architecture uses a linear adapter between encoder...

## How It Works

**References:**

- Model: "Nemotron-Speech" (NVIDIA, 2025)

Nemotron-Speech uses a Fast Conformer encoder paired with a Nemotron LLM decoder for
multi-task speech understanding: ASR, translation, summarization, and instruction-following
on audio input. The architecture uses a linear adapter between encoder and LLM to project
audio representations into the LLM's token embedding space. The model leverages the LLM's
instruction-following capabilities to handle diverse speech tasks via natural language
prompts, producing text output conditioned on both audio content and task instructions.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Fast Conformer encoder + Nemotron LLM decoder. |

