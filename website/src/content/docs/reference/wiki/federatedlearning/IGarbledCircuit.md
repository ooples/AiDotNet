---
title: "IGarbledCircuit"
description: "Defines the contract for garbled circuit generation and evaluation."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.MPC`

Defines the contract for garbled circuit generation and evaluation.

## For Beginners

Garbled circuits are a technique for two parties to compute any
function on their combined inputs without revealing their inputs to each other.

## How It Works

**How it works:**

**Optimizations supported:**

## Methods

| Method | Summary |
|:-----|:--------|
| `Decode(Byte[][],Byte[][])` | Decodes the output wire labels to actual output bits. |
| `Evaluate(GarbledCircuitData,Byte[][])` | Evaluates a garbled circuit given input wire labels. |
| `Garble(IReadOnlyList<CircuitGate>,Int32,Int32)` | Garbles a boolean circuit represented as a list of gate operations. |

