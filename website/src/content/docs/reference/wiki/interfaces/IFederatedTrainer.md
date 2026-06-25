---
title: "IFederatedTrainer<TModel, TData, TMetadata>"
description: "Defines the core functionality for federated learning trainers that coordinate distributed training across multiple clients."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the core functionality for federated learning trainers that coordinate distributed training across multiple clients.

## How It Works

This interface represents the fundamental operations for federated learning systems where multiple clients
(devices, institutions, edge nodes) collaboratively train a shared model without sharing their raw data.

**For Beginners:** Federated learning is like group study where everyone learns from their own materials
but shares only their insights, not their actual study materials.

Think of federated learning as a privacy-preserving collaborative learning approach:

- Multiple clients (hospitals, phones, banks) have their own local data
- Each client trains a model on their local data independently
- Only model updates (not raw data) are shared with a central server
- The server aggregates these updates to improve the global model
- The improved global model is sent back to clients for the next round

For example, in healthcare:

- Multiple hospitals want to train a disease detection model
- Each hospital has patient data that cannot be shared due to privacy regulations
- Each hospital trains the model on their own data
- Only the learned patterns (model weights) are shared and combined
- This creates a better model while keeping patient data private

This interface provides methods for coordinating the federated training process.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetGlobalModel` | Retrieves the current global model after federated training. |
| `Initialize(,Int32)` | Initializes the federated learning process with client configurations and the global model. |
| `SetAggregationStrategy(IAggregationStrategy<>)` | Sets the aggregation strategy used to combine client updates. |
| `Train(Dictionary<Int32,>,Int32,Double,Int32)` | Executes multiple rounds of federated learning until convergence or maximum rounds reached. |
| `TrainRound(Dictionary<Int32,>,Double,Int32)` | Executes one round of federated learning where clients train locally and updates are aggregated. |

