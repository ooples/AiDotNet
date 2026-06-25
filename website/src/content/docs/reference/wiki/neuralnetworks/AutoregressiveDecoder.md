---
title: "AutoregressiveDecoder<T>"
description: "Generic autoregressive decode loop (#1632 / #95): the reusable \"generate\" driver the codebase lacked — GPT4Vision / Blip / Flamingo each hand-rolled this loop."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.NeuralNetworks.Generation`

Generic autoregressive decode loop (#1632 / #95): the reusable "generate" driver the codebase
lacked — GPT4Vision / Blip / Flamingo each hand-rolled this loop. It owns the loop + EOS + token
feedback + RNG lifetime; the model supplies the per-step "embed the token, run the (KV-cached)
forward, return next-token logits" via the `stepLogits` delegate. Because the cached
attention layers append to the KV cache inside that forward, each step only pays for the new
token instead of recomputing the prefix (proven equivalent by KVCacheDecodeEquivalenceTests).

## Methods

| Method | Summary |
|:-----|:--------|
| `Decode(Func<Nullable<Int32>,Vector<>>,Int32,SamplingOptions,Func<Int32,Boolean>,Func<Int32,Boolean>)` | Greedily/stochastically decodes up to `maxNewTokens` tokens. |

