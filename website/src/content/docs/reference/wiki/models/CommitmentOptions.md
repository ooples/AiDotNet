---
title: "CommitmentOptions"
description: "Configuration options for cryptographic commitment schemes."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for cryptographic commitment schemes.

## For Beginners

A commitment scheme lets you "lock in" a value without revealing it.
Later, you can "open" the commitment to prove what the original value was. No one can
change the committed value after the fact (binding), and no one can learn the value
before you open it (hiding).

## Properties

| Property | Summary |
|:-----|:--------|
| `HashAlgorithm` | Gets or sets the hash algorithm for hash-based commitments. |
| `PedersenGroupBitLength` | Gets or sets the Pedersen generator parameter (prime modulus bit length). |
| `RandomnessLength` | Gets or sets the randomness length in bytes for commitment blinding. |
| `UseBatchCommitments` | Gets or sets whether to use batch commitments (commit to multiple values at once). |

