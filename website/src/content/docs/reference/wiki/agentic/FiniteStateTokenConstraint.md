---
title: "FiniteStateTokenConstraint"
description: "A constraint defined by a finite-state grammar over token ids: the set of allowed next tokens depends on the most recently generated token (the current state)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Local`

A constraint defined by a finite-state grammar over token ids: the set of allowed next tokens depends on
the most recently generated token (the current state). This expresses exact sequences, branching choices,
and loops — the general mechanism a JSON-schema or regular grammar compiles down to.

## For Beginners

Picture a flowchart where each box says "from here, you may only go to these
tokens next". Generation walks the flowchart; it can never step off it. Give each box exactly one exit and
you force a precise output; give it several and you allow choices. Boxes with no exit end the answer.

## How It Works

State is the last generated token. Before any token is generated, the `start` set applies. From a
state with no outgoing transitions, the empty set is returned, which tells the engine to stop — so a
chain of single-token transitions forces an exact output, and terminal states end generation cleanly.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FiniteStateTokenConstraint(IEnumerable<Int32>,IReadOnlyDictionary<Int32,IReadOnlyCollection<Int32>>)` | Initializes a new finite-state constraint. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AllowedNextTokens(IReadOnlyList<Int32>)` |  |

