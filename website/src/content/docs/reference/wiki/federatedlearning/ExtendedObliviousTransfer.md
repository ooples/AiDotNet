---
title: "ExtendedObliviousTransfer"
description: "Implements OT extension — amortizes a small number of base OTs into many cheap OTs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.MPC`

Implements OT extension — amortizes a small number of base OTs into many cheap OTs.

## For Beginners

Base oblivious transfer uses expensive public-key cryptography.
OT extension lets you perform a small number of base OTs (e.g., 128) and then "extend" them
into millions of OTs using only symmetric crypto (hashing). This makes garbled circuit
evaluation practical.

## How It Works

**How it works (simplified):**

**Reference:** IKNP OT Extension (Ishai, Kilian, Nissim, Petrank, CRYPTO 2003).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExtendedObliviousTransfer(IObliviousTransfer,Int32)` | Initializes a new instance of `ExtendedObliviousTransfer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseTransferCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BatchTransfer(Byte[][],Byte[][],Int32[])` |  |
| `Initialize` | Initializes the OT extension by running the base OTs. |
| `Transfer(Byte[],Byte[],Int32)` |  |

