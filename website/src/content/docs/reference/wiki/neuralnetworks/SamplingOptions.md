---
title: "SamplingOptions"
description: "Decoding/sampling configuration for autoregressive text generation (#1632 / #95)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.NeuralNetworks.Generation`

Decoding/sampling configuration for autoregressive text generation (#1632 / #95).
Centralises the knobs that were previously reimplemented per multimodal model
(GPT4Vision/Blip/Flamingo each rolled their own temperature + softmax + sample loop).

## How It Works

**For Beginners:** these control how the model picks the next token.

- **Temperature** — randomness. 0 (or very small) = always pick the single most likely

token (greedy, deterministic). 1.0 = sample from the model's raw distribution. >1 = flatter /
more random; <1 = sharper / more focused.

- **TopK** — only consider the K most-likely tokens (0 = no limit).
- **TopP** (nucleus) — only consider the smallest set of tokens whose probabilities sum to

at least P (0 or ≥1 = no limit).

- **Seed** — set for reproducible sampling; null uses the shared thread-safe RNG.

## Properties

| Property | Summary |
|:-----|:--------|
| `Default` | Default sampling (temperature 1.0, no top-k/top-p). |
| `Greedy` | Greedy (argmax) decoding — deterministic. |
| `IsGreedy` | True when these options request deterministic greedy decoding. |
| `Seed` | Seed for reproducible sampling; null ⇒ shared thread-safe RNG. |
| `Temperature` | Softmax temperature. |
| `TopK` | Keep only the K highest-logit tokens before sampling. |
| `TopP` | Nucleus threshold: keep the smallest token set whose cumulative probability ≥ TopP. |

## Fields

| Field | Summary |
|:-----|:--------|
| `GreedyTemperatureEpsilon` | At/below this temperature, sampling collapses to greedy argmax (avoids divide-by-~0). |

