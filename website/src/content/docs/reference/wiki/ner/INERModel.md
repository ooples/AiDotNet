---
title: "INERModel<T>"
description: "Base interface for all Named Entity Recognition (NER) AI models in AiDotNet."
section: "API Reference"
---

`Interfaces` · `AiDotNet.NER.Interfaces`

Base interface for all Named Entity Recognition (NER) AI models in AiDotNet.

## For Beginners

A NER model reads text and highlights important named things in it,
like a highlighter that uses different colors for different types of entities.

For example, given the sentence "Albert Einstein worked at Princeton University in New Jersey":

- "Albert Einstein" is highlighted as a **PERSON** (someone's name)
- "Princeton University" is highlighted as an **ORGANIZATION** (an institution)
- "New Jersey" is highlighted as a **LOCATION** (a place)

The model works by processing each word (token) in the sentence and assigning it a label
from the BIO scheme:

- **B-** prefix means "Beginning of an entity" (e.g., B-PER for the first word of a person name)
- **I-** prefix means "Inside an entity" (e.g., I-PER for subsequent words of a person name)
- **O** means "Outside any entity" (regular words like "worked", "at", "in")

Key technical concepts:

- Input tensors represent token embeddings with shape [batch, sequenceLength, embeddingDim]
- Output tensors represent label predictions with shape [batch, sequenceLength] or label scores
- Models can run in Native mode (pure C# with full training) or ONNX mode (optimized inference)
- All models inherit full serialization, checkpointing, and gradient computation from IFullModel

Example usage:

## How It Works

Named Entity Recognition (NER) is the task of identifying and classifying named entities
in text into predefined categories such as person names, organizations, locations, dates,
and more. This interface extends IFullModel to provide the core contract that all NER models
must implement, inheriting standard methods for training, inference, model persistence,
and gradient computation.

NER is a fundamental building block for many NLP applications:

- **Information extraction:** Pulling structured data from unstructured text
- **Question answering:** Identifying entities mentioned in questions and documents
- **Search engines:** Understanding queries that mention specific people, places, or organizations
- **Knowledge graph construction:** Populating knowledge bases from text corpora
- **Medical NLP:** Extracting drug names, diseases, and symptoms from clinical notes

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the embedding dimension for input token representations. |
| `ExpectedInputShape` | Gets the expected input tensor shape for this model. |
| `NumLabels` | Gets the number of entity label classes this model predicts. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetModelSummary` | Gets a human-readable summary of the model architecture. |
| `PredictBatch(IEnumerable<Tensor<>>)` | Processes multiple sentences in a batch, predicting entity labels for each one. |
| `PredictLabels(Tensor<>)` | Performs NER prediction on input token embeddings and returns the optimal label sequence. |
| `TrainAsync(Tensor<>,Tensor<>,Int32,IProgress<NERTrainingProgress>,CancellationToken)` | Trains the model on labeled NER data asynchronously with progress reporting. |
| `ValidateInputShape(Tensor<>)` | Validates that an input tensor has the correct shape for this model. |

