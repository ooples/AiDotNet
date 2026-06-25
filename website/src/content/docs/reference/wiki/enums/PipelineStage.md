---
title: "PipelineStage"
description: "Defines which stage of an AI pipeline a component operates in."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines which stage of an AI pipeline a component operates in.

## For Beginners

AI systems often process data through multiple stages, like an assembly line.
This tells you where in the pipeline a component fits. Components in the same stage can be
swapped; components in different stages are composed sequentially.

## How It Works

A typical RAG pipeline: DataIngestion → Indexing → Retrieval → PostRetrieval → Generation.
A typical ML pipeline: Preprocessing → Training → Evaluation.

## Fields

| Field | Summary |
|:-----|:--------|
| `DataIngestion` | Data ingestion stage: parsing, chunking, and preparing raw data. |
| `Evaluation` | Evaluation stage: measuring model/pipeline quality. |
| `Generation` | Generation stage: producing final output from retrieved context. |
| `Indexing` | Indexing stage: embedding and storing processed data for retrieval. |
| `PostRetrieval` | Post-retrieval stage: refining retrieved results before generation. |
| `Preprocessing` | Preprocessing stage: transforming raw features before model training/inference. |
| `QueryProcessing` | Query processing stage: transforming user queries before retrieval. |
| `Retrieval` | Retrieval stage: searching and filtering stored data given a query. |
| `Training` | Training stage: optimizing model parameters from data. |

