---
title: "GlooCommunicationBackend<T>"
description: "Gloo-based communication backend for CPU-based collective operations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Gloo-based communication backend for CPU-based collective operations.

## How It Works

**Overview:**
Gloo is Facebook's collective communications library optimized for both CPUs and GPUs.
It provides efficient implementations of collective operations for CPU-based training
or heterogeneous environments. Gloo is particularly well-suited for training on CPUs
or mixed CPU/GPU clusters where NCCL may not be available or optimal.

**Features:**

- CPU-optimized collective operations
- Supports TCP, InfiniBand via ibverbs
- Works on both CPUs and GPUs
- Cross-platform (Linux, macOS, Windows)
- Used by PyTorch's distributed package

**Use Cases:**

- CPU-based distributed training
- Heterogeneous clusters (mixed CPU/GPU)
- When NCCL is not available (non-NVIDIA hardware, macOS, etc.)
- Development and testing on laptops/workstations
- Production training on CPU clusters

**Requirements:**

- Gloo library (C++)
- .NET bindings for Gloo (custom P/Invoke or wrapper library)
- Network connectivity between workers (TCP/IP or InfiniBand)

**Architecture:**
This backend supports two modes of operation:

1. **Native Gloo Mode (Optional):**

Requires GlooSharp package (separate NuGet) which provides .NET bindings for the
native Gloo C++ library. Gloo offers optimized collective operations for CPU and GPU.
To use: Install the GlooSharp package separately.

2. **Built-in TCP Mode (Default, Production-Ready):**

Production-ready TCP-based implementation using industry-standard ring algorithms
(ring-allreduce, ring-allgather, ring-reduce-scatter). Provides full multi-process
functionality without external dependencies.

The TCP implementation features:

- Automatic TCP connection initialization with retry logic and handshakes
- Ring-based collective operations for optimal bandwidth utilization
- Proper error handling, validation, and timeout mechanisms
- Environment-based rendezvous (AIDOTNET_MASTER_ADDR, AIDOTNET_MASTER_PORT)
- Support for arbitrary world sizes and fault-tolerant connection establishment

**Environment Variables:**

- AIDOTNET_GLOO_TRANSPORT: Transport to use ("tcp" or "ib"/"infiniband"). Default: "tcp".
- AIDOTNET_GLOO_IB_DEVICE: InfiniBand device name (only when transport is "ib").
- AIDOTNET_GLOO_STORE_PATH: Filesystem path for Gloo rendezvous/store coordination.
- AIDOTNET_MASTER_ADDR: IP address of rank 0 for TCP rendezvous.
- AIDOTNET_MASTER_PORT: Base port number for TCP connections (each rank uses port + rank).

**Recommendation:** Use TCP mode for most scenarios. Add GlooSharp only if you need
specialized hardware support (InfiniBand) or have specific Gloo optimizations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GlooCommunicationBackend(Int32,Int32)` | Creates a new Gloo communication backend using production-ready TCP implementation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Rank` |  |
| `WorldSize` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AcceptConnectionFromAnyRank` | Accepts connection from any rank (passive connection). |
| `AllGather(Vector<>)` |  |
| `AllReduce(Vector<>,ReductionOperation)` |  |
| `Barrier` |  |
| `Broadcast(Vector<>,Int32)` |  |
| `CleanupOnInitFailure` | Cleans up all resources (native Gloo context + TCP connections) on initialization failure. |
| `ConnectToRank(Int32,String,Int32)` | Connects to a specific rank (active connection). |
| `CreateGlooReduceOpMap(Type)` | Creates a cached mapping from `ReductionOperation` to GlooSharp GlooReduceOp enum values. |
| `DoubleArrayToVector(Double[])` | Converts a `double[]` back to a `Vector`. |
| `InitializeTCPConnections` | Initializes TCP connections between all ranks. |
| `MapToGlooReduceOp(ReductionOperation)` | Maps an AiDotNet `ReductionOperation` to the cached GlooSharp GlooReduceOp enum value. |
| `OnInitialize` |  |
| `OnShutdown` |  |
| `PerformNativeGlooAllGather(Vector<>)` | Performs AllGather via native Gloo (GlooSharp). |
| `PerformNativeGlooAllReduce(Vector<>,ReductionOperation)` | Performs AllReduce via native Gloo (GlooSharp). |
| `PerformNativeGlooBroadcast(Vector<>,Int32)` | Performs Broadcast via native Gloo (GlooSharp). |
| `PerformNativeGlooReduceScatter(Vector<>,ReductionOperation)` | Performs ReduceScatter via native Gloo (GlooSharp). |
| `PerformReduction(,,ReductionOperation)` | Performs a reduction operation on two values. |
| `PerformRingAllGather(Vector<>)` | Performs ring-based AllGather operation. |
| `PerformRingAllReduce(Vector<>,ReductionOperation)` | Performs ring-based AllReduce operation. |
| `PerformRingReduceScatter(Vector<>,ReductionOperation)` | Performs ring-based ReduceScatter operation. |
| `PerformTreeBroadcast(Vector<>,Int32)` | Performs tree-based Broadcast operation. |
| `PerformTreeScatter(Vector<>,Int32)` | Performs tree-based Scatter operation. |
| `Receive(Int32,Int32,Int32)` |  |
| `ReceiveData(Int32,Int32)` | Receives data from a specific rank via TCP. |
| `ReceiveDataWithTag(Int32,Int32,Int32)` | Receives data from a specific rank via TCP with message tag. |
| `ReduceScatter(Vector<>,ReductionOperation)` |  |
| `Scatter(Vector<>,Int32)` |  |
| `ScatterTreeSend([],Int32,Int32,Int32,Int32)` | Recursive helper for tree-based scatter send. |
| `Send(Vector<>,Int32,Int32)` |  |
| `SendData(Int32,Vector<>)` | Sends data to a specific rank via TCP. |
| `SendDataWithTag(Int32,Vector<>,Int32)` | Sends data to a specific rank via TCP with message tag. |
| `TryConfigureNativeGloo(IDisposable,Type,Type)` | Configures native Gloo transport, caches method handles, and transfers ownership of context on success. |
| `TryInitializeNativeGloo` | Attempts to detect and initialize the GlooSharp native package via reflection. |
| `VectorToDoubleArray(Vector<>)` | Converts a `Vector` to a `double[]` for native Gloo calls. |

