---
title: "SecureAggregation<T>"
description: "Implements secure aggregation for federated learning using cryptographic techniques."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Privacy`

Implements secure aggregation for federated learning using cryptographic techniques.

## How It Works

Secure aggregation is a cryptographic protocol that allows a server to compute the sum
of client updates without seeing individual contributions. Only the final aggregate is
visible to the server.

**For Beginners:** Secure aggregation is like a secret ballot election where votes
are counted but individual votes remain private.

How it works (simplified):

1. Each client generates pairwise secret keys with other clients
2. Clients mask their model updates with these secret keys
3. Server receives masked updates: masked_update_i = update_i + Σ(secrets_ij)
4. Secret masks cancel out when summing: Σ(masked_update_i) = Σ(update_i)
5. Server gets the sum without seeing individual updates

Example with 3 clients:

- Client 1 shares secrets: s₁₂ with Client 2, s₁₃ with Client 3
- Client 2 shares secrets: s₂₁ with Client 1, s₂₃ with Client 3
- Client 3 shares secrets: s₃₁ with Client 1, s₃₂ with Client 2

Note: s₁₂ = -s₂₁ (secrets cancel in pairs)

Client 1 sends: update₁ + s₁₂ + s₁₃
Client 2 sends: update₂ + s₂₁ + s₂₃
Client 3 sends: update₃ + s₃₁ + s₃₂

Server computes sum:
(update₁ + s₁₂ + s₁₃) + (update₂ + s₂₁ + s₂₃) + (update₃ + s₃₁ + s₃₂)
= update₁ + update₂ + update₃ + (s₁₂ + s₂₁) + (s₁₃ + s₃₁) + (s₂₃ + s₃₂)
= update₁ + update₂ + update₃ + 0 + 0 + 0
= Σ(updates) ← Only this is visible to server!

This implementation derives pairwise mask seeds from per-round ephemeral ECDH shared secrets and expands them
via HKDF + a deterministic PRG. Pairwise masks cancel in the aggregate as long as all selected clients participate
in the round (synchronous, full-participation mode).

Benefits:

- Server cannot see individual client updates
- Protects against honest-but-curious server
- No trusted third party needed
- Computation overhead is reasonable

Limitations:

- Requires coordination between clients
- All (or threshold) clients must participate for masks to cancel
- Dropout handling requires additional mechanisms
- Communication overhead for key exchange

When to use Secure Aggregation:

- Don't fully trust the central server
- Regulatory requirements for data protection
- Want cryptographic privacy guarantees
- Willing to handle additional complexity

Can be combined with differential privacy for stronger protection:

- Secure aggregation: Protects individual updates from server
- Differential privacy: Protects individual data points from anyone

Reference: Bonawitz, K., et al. (2017). "Practical Secure Aggregation for Privacy-Preserving
Machine Learning." CCS 2017.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SecureAggregation(Int32,Nullable<Int32>)` | Initializes a new instance of the `SecureAggregation` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateSecurely(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` | Aggregates masked updates from all clients, returning a weighted average. |
| `AggregateSumSecurely(Dictionary<Int32,Dictionary<String,[]>>)` | Aggregates masked updates from all clients, returning the raw sum with masks cancelled. |
| `ClearSecrets` | Clears all stored pairwise secrets. |
| `FlattenParameters(Dictionary<String,[]>)` | Flattens a hierarchical model structure into a single parameter array. |
| `GeneratePairwiseSecrets(List<Int32>)` | Generates pairwise secrets between all clients. |
| `GetClientCount` | Gets the number of clients with stored secrets. |
| `MaskUpdate(Int32,Dictionary<String,[]>)` | Masks a client's model update with pairwise secrets. |
| `MaskUpdate(Int32,Dictionary<String,[]>,Double)` | Masks a client's model update with pairwise secrets, applying the client's aggregation weight before masking. |

