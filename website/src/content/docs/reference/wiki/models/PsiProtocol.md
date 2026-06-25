---
title: "PsiProtocol"
description: "Specifies the cryptographic protocol used for Private Set Intersection."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the cryptographic protocol used for Private Set Intersection.

## For Beginners

Different PSI protocols trade off speed, security, and functionality.
Think of these as different methods of comparing guest lists privately:

## Fields

| Field | Summary |
|:-----|:--------|
| `BloomFilter` | Bloom filter based probabilistic PSI. |
| `CircuitBased` | Circuit-based PSI using garbled circuits or secret sharing. |
| `DiffieHellman` | Diffie-Hellman based PSI using commutative encryption. |
| `ObliviousTransfer` | Oblivious Transfer based PSI using cuckoo hashing. |

