---
title: "SSLCommunicationBackend"
description: "Communication backends for distributed SSL training."
section: "API Reference"
---

`Enums` · `AiDotNet.SelfSupervisedLearning`

Communication backends for distributed SSL training.

## Fields

| Field | Summary |
|:-----|:--------|
| `Gloo` | Gloo backend (CPU or fallback). |
| `InMemory` | In-memory communication (single machine, testing). |
| `MPI` | MPI backend (multi-node, HPC clusters). |
| `NCCL` | NCCL backend (NVIDIA GPUs, best for multi-GPU). |

