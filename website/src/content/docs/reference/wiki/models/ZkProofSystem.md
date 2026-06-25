---
title: "ZkProofSystem"
description: "Specifies the zero-knowledge proof system to use for verifiable federated learning."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the zero-knowledge proof system to use for verifiable federated learning.

## For Beginners

Zero-knowledge proofs let you prove something is true without
revealing why it's true. Different proof systems trade off between speed, proof size,
and what they can prove:

## Fields

| Field | Summary |
|:-----|:--------|
| `Bulletproofs` | Bulletproofs — logarithmic-size range proofs without trusted setup. |
| `Groth16` | Groth16 — constant-size proofs for arbitrary computations. |
| `HashCommitment` | SHA-256 hash commitment — fast, simple, non-interactive. |
| `Pedersen` | Pedersen commitment — additively homomorphic. |
| `Plonk` | PLONK — universal setup, supports arbitrary computations. |

