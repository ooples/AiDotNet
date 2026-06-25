---
title: "GarbledCircuitEvaluator"
description: "Evaluates garbled circuits produced by `GarbledCircuitGenerator`."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.MPC`

Evaluates garbled circuits produced by `GarbledCircuitGenerator`.

## For Beginners

The evaluator is the second party in a garbled circuit protocol.
It receives the garbled circuit and wire labels for its inputs (via oblivious transfer),
then processes each gate to compute the output — without learning the garbler's inputs
or any intermediate values.

## How It Works

**Workflow:**

**Security:** The evaluator learns only the output. It cannot determine the garbler's
inputs because each wire label is a random cryptographic value that reveals nothing about
the underlying bit.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GarbledCircuitEvaluator(Boolean,Int32)` | Initializes a new instance of `GarbledCircuitEvaluator`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Decode(Byte[][],Byte[][])` |  |
| `Evaluate(GarbledCircuitData,Byte[][])` |  |
| `Garble(IReadOnlyList<CircuitGate>,Int32,Int32)` | Not supported by the evaluator — use `GarbledCircuitGenerator` to garble. |
| `ObtainInputLabels(GarbledCircuitData,Int32[],Int32,IObliviousTransfer)` | Obtains input wire labels for the evaluator's input bits using oblivious transfer. |

