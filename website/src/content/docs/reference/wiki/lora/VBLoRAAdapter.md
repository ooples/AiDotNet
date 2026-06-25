---
title: "VBLoRAAdapter<T>"
description: "Vector Bank LoRA (VB-LoRA) adapter that uses shared parameter banks for efficient multi-client deployment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LoRA.Adapters`

Vector Bank LoRA (VB-LoRA) adapter that uses shared parameter banks for efficient multi-client deployment.

## For Beginners

Think of vector banks like a shared library of building blocks.

Traditional LoRA is like each person having their own complete toolbox:

- Person 1: Full set of tools
- Person 2: Full set of tools
- Person 3: Full set of tools

Result: Lots of duplicate tools

VB-LoRA is like a shared tool library:

- Central tool bank: One of each tool type
- Each person: List of which tools they need (indices)
- Everyone shares the same physical tools

Result: Much fewer tools needed overall

This is especially powerful when many adapters need similar adjustments (common in
multi-task learning or personalization scenarios).

## How It Works

VB-LoRA (2024) introduces vector banks - reusable parameter stores shared across multiple LoRA adapters.
Instead of each adapter having its own complete A and B matrices, VB-LoRA maintains global banks of
column vectors. Each adapter selects which vectors from the banks to use via index arrays.

**Key Innovation:** Vector Bank Architecture

Traditional LoRA:

- Each adapter stores full A (inputSize × rank) and B (rank × outputSize) matrices
- Total parameters for N adapters = N × (inputSize × rank + rank × outputSize)
- No sharing between adapters

VB-LoRA:

- Global BankA contains pooled column vectors (inputSize × bankSize)
- Global BankB contains pooled column vectors (bankSize × outputSize)
- Each adapter stores only indices (which vectors to use from banks)
- Total parameters = (inputSize × bankSize + bankSize × outputSize) + N × 2 × rank × sizeof(int)
- Massive reduction when bankSize << N × rank

**Benefits:**

1. **Reduced Duplication**: Similar adapters share vector bank entries
2. **Lower Communication Overhead**: Multi-client systems can cache banks locally
3. **Memory Efficiency**: Fewer unique parameters to store and transmit
4. **Scalability**: Adding new adapters only requires index arrays, not full matrices
5. **Knowledge Sharing**: Banks capture common adaptation patterns

**Example Scenario:**

Suppose you're deploying personalized language models to 1000 users:

- Traditional LoRA: Each user needs their own 16K parameter adapter (16MB total)
- VB-LoRA: Shared 256K parameter bank + 1000 users × 128 indices each
- Result: 84% memory reduction (256K + 128K vs 16M)

The shared bank captures common language patterns, while per-user indices
select the patterns relevant to each individual.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VBLoRAAdapter(ILayer<>,Int32,Int32,Int32,Int32[],Int32[],Double,Boolean,String)` | Initializes a new VB-LoRA adapter with specified bank sizes and indices. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BankIndicesA` | Gets the indices into Bank A used by this adapter. |
| `BankIndicesB` | Gets the indices into Bank B used by this adapter. |
| `BankSizeA` | Gets the size of Bank A (number of available column vectors). |
| `BankSizeB` | Gets the size of Bank B (number of available row vectors). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearBanks(String)` | Clears the global vector banks (useful for testing or reinitialization). |
| `CreateLoRALayer(Int32,Double)` | Creates the LoRA layer for this adapter, customized to use vector bank indices. |
| `Forward(Tensor<>)` | Performs the forward pass using bank-selected vectors. |
| `GenerateRandomIndices(Int32,Int32)` | Generates random indices for bank vector selection. |
| `GetBankA(String)` | Gets the global vector bank A for inspection or advanced use cases. |
| `GetBankB(String)` | Gets the global vector bank B for inspection or advanced use cases. |
| `InitializeBanksIfNeeded(Int32,Int32)` | Initializes the vector banks if they don't exist for this bank key. |
| `MergeToOriginalLayer` | Merges the VB-LoRA adaptation into the base layer and returns the merged layer. |
| `UpdateBanksFromLoRALayer(LoRALayer<>)` | Writes the LoRA layer's updated parameters back to the shared banks. |
| `UpdateLoRALayerFromBanks(LoRALayer<>)` | Updates the LoRA layer's matrices by extracting selected vectors from the banks. |
| `UpdateParameters()` | Updates parameters and propagates changes back to the shared banks. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bankIndicesA` | Indices into BankA - specifies which column vectors from the bank to use for this adapter. |
| `_bankIndicesB` | Indices into BankB - specifies which row vectors from the bank to use for this adapter. |
| `_bankKey` | Unique identifier for the bank configuration (used as dictionary key). |
| `_bankLock` | Lock object for thread-safe bank initialization and access. |
| `_bankSizeA` | Size of the vector bank A (number of available column vectors). |
| `_bankSizeB` | Size of the vector bank B (number of available row vectors). |
| `_globalBankA` | Global bank of column vectors for matrix A, shared across all VB-LoRA instances. |
| `_globalBankB` | Global bank of column vectors for matrix B, shared across all VB-LoRA instances. |

