---
title: "ICausalLanguageModel<T>"
description: "The minimal contract an in-process language model exposes to the local generation engine: given the tokens seen so far, produce the logits for the next token."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.Models.Local`

The minimal contract an in-process language model exposes to the local generation engine: given the
tokens seen so far, produce the logits for the next token. This is the seam between AiDotNet's own
Transformer (or any other model) and `LocalEngineChatClient`.

## For Beginners

A language model, at its heart, answers one question over and over: "given
everything so far, how likely is each possible next word-piece?" Those likelihoods (before turning them
into probabilities) are called *logits*. This interface is exactly that one question, so the rest
of the engine can focus on *choosing* the next token and stitching the words back together.

## How It Works

Keeping the contract this small means the generation loop, sampling, and chat templating are written
once and tested independently of any particular model, and the real network is wired in behind this
interface. An implementation is free to maintain an internal KV-cache keyed on the growing context so
repeated calls stay efficient; callers only ever ask for "the next-token logits given this context".

## Properties

| Property | Summary |
|:-----|:--------|
| `VocabularySize` | Gets the size of the model's vocabulary (the length of the logits vector returned by `Int32})`). |

## Methods

| Method | Summary |
|:-----|:--------|
| `NextTokenLogits(IReadOnlyList<Int32>)` | Computes the next-token logits for the given context. |

