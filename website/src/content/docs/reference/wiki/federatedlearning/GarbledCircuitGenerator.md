---
title: "GarbledCircuitGenerator"
description: "Implements Yao's garbled circuit generation with point-and-permute, free XOR, and half-gates optimizations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.MPC`

Implements Yao's garbled circuit generation with point-and-permute, free XOR, and half-gates optimizations.

## For Beginners

A garbled circuit lets two parties compute any function on their
combined inputs without revealing those inputs to each other. The "garbler" takes a boolean
circuit (made of AND, XOR, NOT gates) and "garbles" it — replacing each wire's 0/1 values
with random cryptographic labels. The "evaluator" can then process the circuit using only
the labels for its inputs (obtained via oblivious transfer) without learning anything else.

## How It Works

**Optimizations implemented:**

**Reference:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GarbledCircuitGenerator(Boolean,Boolean,Int32)` | Initializes a new instance of `GarbledCircuitGenerator`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Decode(Byte[][],Byte[][])` |  |
| `Evaluate(GarbledCircuitData,Byte[][])` |  |
| `Garble(IReadOnlyList<CircuitGate>,Int32,Int32)` |  |

