---
title: "IObliviousTransfer"
description: "Defines the contract for an oblivious transfer (OT) protocol."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.MPC`

Defines the contract for an oblivious transfer (OT) protocol.

## For Beginners

Oblivious transfer is a fundamental cryptographic building block.
A sender has two messages (m0 and m1). A receiver has a choice bit (0 or 1). After the
protocol:

## How It Works

**Why this matters for FL:** OT is used as a building block for garbled circuits
and general MPC. When evaluating a garbled circuit, the evaluator needs to obtain the
correct wire labels for its input bits without revealing those bits to the garbler.

**Performance:** Base OT is expensive (uses public-key crypto). OT extension lets
you amortize a small number of base OTs into many cheap OTs using symmetric crypto only.

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseTransferCount` | Gets the number of base OTs this protocol has performed (for accounting/extension). |

## Methods

| Method | Summary |
|:-----|:--------|
| `BatchTransfer(Byte[][],Byte[][],Int32[])` | Performs a batch of 1-out-of-2 oblivious transfers. |
| `Transfer(Byte[],Byte[],Int32)` | Performs a 1-out-of-2 oblivious transfer. |

