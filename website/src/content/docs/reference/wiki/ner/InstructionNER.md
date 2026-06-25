---
title: "InstructionNER<T>"
description: "InstructionNER: Instruction-tuned transformer for few-shot and zero-shot NER via natural language instructions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

InstructionNER: Instruction-tuned transformer for few-shot and zero-shot NER via natural
language instructions.

## For Beginners

InstructionNER works like giving a smart assistant a task description:
"Find all the company names in this text." The model reads the instruction and the text,
then extracts the relevant entities. Because it understands instructions in plain English,
it can adapt to new entity types just by changing the instruction, without retraining.

## How It Works

InstructionNER (Wang et al., ACL 2022 - "InstructionNER: A Multi-Task Instruction-Based
Generative Framework for Few-shot NER") uses instruction tuning to enable NER through
natural language task descriptions, enabling strong few-shot and zero-shot performance.

**Key Innovation - Instruction Tuning for NER:**
Instead of training a model to predict BIO labels, InstructionNER trains the model to
follow natural language instructions that describe the NER task. This leverages the
instruction-following capabilities of large language models.

**Instruction Format:****Multi-Task Training:**
InstructionNER is trained on multiple NER datasets simultaneously with different
instruction prompts, teaching the model to:

1. Extract entities of specified types from text
2. Classify given spans into entity types
3. Verify whether a span is a valid entity of a given type
4. Generate all entities of a specific type

**Few-Shot and Zero-Shot Performance:**

- Zero-shot (new entity types): ~50-60% F1 depending on similarity to training types
- 5-shot: ~65-75% F1
- 20-shot: ~75-85% F1
- Full training: ~93.0% F1 on CoNLL-2003

**Cross-Domain Transfer:**
Because instructions describe the task in natural language, InstructionNER can transfer
to new domains and entity types that were not seen during training. For example, a model
trained on general NER (PER, ORG, LOC) can be prompted to extract domain-specific entities
(drug names, gene names) by simply changing the instruction.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InstructionNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates an InstructionNER model in ONNX inference mode. |
| `InstructionNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an InstructionNER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

