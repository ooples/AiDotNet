---
title: "FederatedAsyncMode"
description: "Specifies the asynchronous federated learning mode."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the asynchronous federated learning mode.

## How It Works

**For Beginners:** In async federated learning, clients can send updates at different times.
The server mixes updates as they arrive instead of waiting for a strict round barrier.

## Fields

| Field | Summary |
|:-----|:--------|
| `AsyncFedED` | AsyncFedED: entropy-driven client scheduling prioritizing most informative clients. |
| `FedAsync` | FedAsync-style staleness-aware mixing. |
| `FedBuff` | FedBuff-style buffered aggregation. |
| `None` | Disable asynchronous modes (standard synchronous rounds). |
| `SemiAsync` | Semi-Async: hybrid sync/async with periodic barriers every K rounds. |

