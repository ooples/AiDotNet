---
title: "IClientModel<TData, TUpdate>"
description: "Defines the functionality for a client-side model in federated learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the functionality for a client-side model in federated learning.

## How It Works

This interface represents a model that exists on a client device or node in a federated
learning system. Each client maintains its own copy of the global model and trains it
on local data.

**For Beginners:** A client model is like a student's personal copy of study materials.
Each student (client) has their own copy, studies it with their own resources, and
contributes improvements back to the class.

Think of client models as distributed learners:

- Each client has a copy of the global model
- Clients train on their own private data
- Local training happens independently and in parallel
- Only model updates (not data) are sent to the server

For example, in smartphone keyboard prediction:

- Each phone has a copy of the global typing prediction model
- The phone learns from the user's typing patterns
- It sends model improvements (not actual typed text) to the server
- The server combines improvements from millions of phones
- Each phone gets the improved model back

This design ensures:

- Data privacy: Raw data never leaves the client
- Personalization: Can adapt to local data distribution
- Scalability: Training happens in parallel across all clients

## Methods

| Method | Summary |
|:-----|:--------|
| `GetModelUpdate` | Computes and retrieves the model update to send to the server. |
| `GetSampleCount` | Gets the number of training samples available on this client. |
| `TrainLocal(,Int32,Double)` | Trains the local model on the client's private data. |
| `UpdateFromGlobal()` | Updates the local model with the new global model from the server. |

