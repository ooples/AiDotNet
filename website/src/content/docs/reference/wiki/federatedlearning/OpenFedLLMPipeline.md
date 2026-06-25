---
title: "OpenFedLLMPipeline<T>"
description: "Implements OpenFedLLM pipeline patterns for federated LLM training, alignment, and serving."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Alignment`

Implements OpenFedLLM pipeline patterns for federated LLM training, alignment, and serving.

## For Beginners

Training a useful LLM happens in stages: first you teach it to
follow instructions (instruction tuning), then you align it with human values (RLHF/DPO),
and finally you deploy it (serving). OpenFedLLM defines how to do each stage in a federated
setting where data stays private. This class orchestrates the three stages and manages
the transitions between them.

## How It Works

Pipeline stages:

Reference: Ye, J., et al. (2024). "OpenFedLLM: Training Large Language Models on
Decentralized Private Data via Federated Learning." arXiv:2402.06954.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OpenFedLLMPipeline(OpenFedLLMOptions)` | Creates a new OpenFedLLM pipeline. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentRound` | Gets the current round number within the current stage. |
| `CurrentStage` | Gets the current pipeline stage. |
| `LastCheckpoint` | Gets a defensive copy of the last checkpointed model parameters, or null if no checkpoint exists. |
| `Options` | Gets the pipeline configuration options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdvanceStage` | Advances to the next pipeline stage, resetting the round counter. |
| `AggregateAlignment(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` | Aggregates alignment-stage model parameters, using either DPO or RLHF aggregation based on the pipeline configuration. |
| `AggregateInstructionTuning(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` | Aggregates instruction-tuned adapter parameters from clients. |
| `IsStageComplete` | Checks whether the current stage has completed its allocated rounds. |
| `RunRound(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` | Runs one round of the pipeline, dispatching to the appropriate stage-specific aggregation. |
| `SaveCheckpoint(Dictionary<String,[]>)` | Saves a checkpoint of the current model parameters. |

