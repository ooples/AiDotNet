---
title: "PipelineStageAttribute"
description: "Specifies which stage(s) of an AI pipeline a component operates in."
section: "API Reference"
---

`Attributes` · `AiDotNet.Attributes`

Specifies which stage(s) of an AI pipeline a component operates in.

## For Beginners

This tells you where in the data processing pipeline a component fits.
A retriever operates in the Retrieval stage, a chunker in DataIngestion, a reranker in
PostRetrieval. Some components span multiple stages.

## How It Works

**Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PipelineStageAttribute(PipelineStage)` | Initializes a new instance of the `PipelineStageAttribute` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Stage` | Gets the pipeline stage this component operates in. |

