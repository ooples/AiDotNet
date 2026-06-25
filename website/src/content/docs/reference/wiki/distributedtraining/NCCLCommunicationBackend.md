---
title: "NCCLCommunicationBackend<T>"
description: "NVIDIA NCCL-based communication backend for GPU-to-GPU communication."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

NVIDIA NCCL-based communication backend for GPU-to-GPU communication.

## How It Works

**Overview:**
NCCL (NVIDIA Collective Communications Library) is optimized for multi-GPU communication
on NVIDIA GPUs. It provides highly optimized implementations of collective operations
that take advantage of NVLink, PCIe, and network topology for maximum throughput.

**Features:**

- Optimized for NVIDIA GPUs (NVLink, NVSwitch awareness)
- Near-optimal bandwidth utilization
- Supports multi-node multi-GPU configurations
- Ring and tree algorithms for different collective operations
- Essential for high-performance multi-GPU training

**Architecture:**
This backend supports two modes of operation:

1. **Native NCCL Mode:**

Uses NCCL library with actual GPU memory for collective operations.
Requires CUDA toolkit and NCCL library. Provides near-optimal GPU bandwidth.

2. **CPU Fallback Mode:**

When NCCL/CUDA not available, uses TCP-based ring algorithms similar to Gloo.
Allows development and testing on systems without NVIDIA GPUs.

The implementation features:

- Automatic NCCL detection and initialization
- TCP-based unique ID distribution for multi-node setup
- Environment-based rendezvous (AIDOTNET_MASTER_ADDR, AIDOTNET_MASTER_PORT)
- Proper CUDA stream synchronization
- Memory-efficient GPU operations

**Requirements for GPU Mode:**

- NVIDIA GPUs (compute capability 3.0+)
- CUDA toolkit 10.0+
- NCCL library 2.0+

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NCCLCommunicationBackend(Int32,Int32,Int32)` | Creates a new NCCL communication backend. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Rank` |  |
| `WorldSize` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AllGather(Vector<>)` |  |
| `AllReduce(Vector<>,ReductionOperation)` |  |
| `Barrier` |  |
| `Broadcast(Vector<>,Int32)` |  |
| `BroadcastUniqueIdTcp(NcclUniqueId)` | Broadcasts NCCL unique ID from rank 0 to all other ranks. |
| `InitializeMultiProcessNCCL` | Initializes NCCL for multi-process mode with TCP-based unique ID distribution. |
| `InitializeNCCL` | Initializes NCCL communicator with proper multi-process setup. |
| `InitializeSingleProcessNCCL` | Initializes NCCL for single-process mode. |
| `InitializeTCPConnections` | Initializes TCP connections for multi-process communication. |
| `OnInitialize` |  |
| `OnShutdown` |  |
| `PerformNcclAllGather(Vector<>)` | Performs AllGather using NCCL with GPU memory. |
| `PerformNcclAllReduce(Vector<>,ReductionOperation)` | Performs AllReduce using NCCL with GPU memory. |
| `PerformNcclBroadcast(Vector<>,Int32)` | Performs Broadcast using NCCL with GPU memory. |
| `PerformNcclReduceScatter(Vector<>,ReductionOperation)` | Performs ReduceScatter using NCCL with GPU memory. |
| `Receive(Int32,Int32,Int32)` |  |
| `ReceiveUniqueIdTcp` | Receives NCCL unique ID from rank 0. |
| `ReduceScatter(Vector<>,ReductionOperation)` |  |
| `Scatter(Vector<>,Int32)` |  |
| `Send(Vector<>,Int32,Int32)` |  |

