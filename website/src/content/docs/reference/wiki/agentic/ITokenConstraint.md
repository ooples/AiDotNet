---
title: "ITokenConstraint"
description: "Restricts which tokens the local engine may generate next, enabling *constrained decoding*: the model can only emit tokens the constraint permits, so the output is guaranteed to satisfy a structure (a fixed vocabulary, a grammar, a JSON sha…"
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.Models.Local`

Restricts which tokens the local engine may generate next, enabling *constrained decoding*: the
model can only emit tokens the constraint permits, so the output is guaranteed to satisfy a structure
(a fixed vocabulary, a grammar, a JSON shape) rather than merely being asked to in the prompt.

## For Beginners

Normally a model can pick any next word-piece. A constraint is a gate that
only lets through the choices that keep the answer valid — for example, only digits when you want a
number, or only tokens that continue well-formed JSON. Because the gate is applied while generating, the
result is always valid by construction, not just "usually" valid.

## How It Works

At each step the engine calls `Int32})` with the tokens generated so far. Returning
`null` means "no restriction this step"; returning a set restricts sampling to exactly those token
ids; returning an empty set tells the engine to stop (nothing valid can follow). This is the foundation
for local structured output and tool-calling — capabilities cloud models approximate with prompting but
cannot guarantee, and which a local engine can enforce at the logits because it controls decoding.

## Methods

| Method | Summary |
|:-----|:--------|
| `AllowedNextTokens(IReadOnlyList<Int32>)` | Returns the token ids permitted as the next token given what has been generated so far. |

