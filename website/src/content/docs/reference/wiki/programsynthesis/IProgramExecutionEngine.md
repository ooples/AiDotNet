---
title: "IProgramExecutionEngine"
description: "Defines an execution boundary for running synthesized programs against inputs."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ProgramSynthesis.Interfaces`

Defines an execution boundary for running synthesized programs against inputs.

## For Beginners

This is the "runner" that actually executes the generated code.

Program synthesis can generate code as text, but to verify it works we need to run it safely.
This interface lets you plug in a safe execution environment (for example, a container,
an isolated process, or a remote service) without embedding unsafe execution inside the library.

## How It Works

Implementations should execute code in a sandboxed, resource-limited environment appropriate
for the target language (timeouts, memory limits, restricted I/O, etc.).

## Methods

| Method | Summary |
|:-----|:--------|
| `TryExecute(ProgramLanguage,String,String,String,String,CancellationToken)` | Tries to execute the given program source against the provided input. |

