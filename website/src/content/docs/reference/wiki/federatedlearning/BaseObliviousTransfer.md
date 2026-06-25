---
title: "BaseObliviousTransfer"
description: "Implements base 1-out-of-2 oblivious transfer using symmetric cryptography."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.MPC`

Implements base 1-out-of-2 oblivious transfer using symmetric cryptography.

## For Beginners

In oblivious transfer (OT), a sender has two messages (m0, m1)
and a receiver has a choice bit (0 or 1). After the protocol:

## How It Works

This implementation uses a simplified random-oracle model where the sender and
receiver derive keys from a shared random seed (simulating the Diffie-Hellman key
exchange that would happen in a real network protocol). In production, this would use
actual public-key cryptography over a network channel.

**Performance:** Each base OT requires public-key operations. Use
`ExtendedObliviousTransfer` to amortize this cost for many transfers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BaseObliviousTransfer` | Initializes a new instance of `BaseObliviousTransfer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseTransferCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BatchTransfer(Byte[][],Byte[][],Int32[])` |  |
| `Transfer(Byte[],Byte[],Int32)` |  |

