---
title: "ComputationIntegrityProof<T>"
description: "Generates proofs of local training computation integrity (research-stage)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Verification`

Generates proofs of local training computation integrity (research-stage).

## For Beginners

This is the "holy grail" of verifiable FL: proving that a client
actually ran the correct training algorithm (e.g., SGD for N epochs on its dataset) without
revealing the data or the model weights.

## How It Works

**How it works (conceptually):**

**Current limitations:** Full computation proofs for deep neural networks are extremely
expensive. Current research (zkRNN 2026, ZKML Survey 2025) shows:

This implementation provides a simplified proof-of-concept using hash chains to verify
training step ordering, not the full SNARK-based approach.

**Reference:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ComputationIntegrityProof(IZkProofSystem,Int32)` | Initializes a new instance of `ComputationIntegrityProof`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ProofSystemName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateProof(Byte[],VerificationConstraint)` |  |
| `GenerateTrainingProof(TrainingStepLog,Int32)` | Generates a proof of training integrity from a training log. |
| `RecordTrainingStep(TrainingStepLog,Byte[],Double,Int32)` | Records a training step by hashing the current state (model hash + loss + epoch). |
| `Verify(VerificationProof,VerificationConstraint)` |  |

