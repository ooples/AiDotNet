---
title: "SequencePoolingMode"
description: "Strategy for collapsing a transformer encoder's `[batch, seq, dim]` hidden states into a single `[batch, dim]` vector before the classification head, when the task is single-label per sequence."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Strategy for collapsing a transformer encoder's `[batch, seq, dim]`
hidden states into a single `[batch, dim]` vector before the
classification head, when the task is single-label per sequence.

## How It Works

Picking the wrong mode silently destroys the position-specific signal
the model needs to learn. The default `LastToken` matches
canonical autoregressive language modelling (GPT-style next-token
prediction): the last position has attended to every preceding
position via causal self-attention and is therefore the natural
summary of the prefix.

`MeanPool` averages over all positions and is appropriate
for non-causal sequence-classification (e.g. document sentiment
where the whole sequence is observed at once and word order matters
less than the overall content). Using MeanPool for next-token LM
produced the flat-softmax convergence failure tracked in
AiDotNet#1232: every context mapped to roughly the same averaged
hidden state, the model couldn't learn position-conditioned outputs,
and softmax converged to `~uniform / V`.

`ClsToken` matches BERT-style models that prepend a
dedicated `[CLS]` token and use its final-layer hidden state
as the sequence summary. `None` keeps the full
`[batch, seq, dim]` shape — the right choice when the loss is
applied per-token (token-classification, masked-LM with parallel
position prediction, sequence-to-sequence training).

## Fields

| Field | Summary |
|:-----|:--------|
| `ClsToken` | Use the FIRST position's hidden state, treated as a prepended `[CLS]` summary token (BERT-style). |
| `LastToken` | Take the LAST position's hidden state. |
| `MeanPool` | Average all positions. |
| `None` | Skip pooling entirely. |

