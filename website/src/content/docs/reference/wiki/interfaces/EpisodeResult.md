---
title: "EpisodeResult<T>"
description: "Result of running a single RL episode."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Result of running a single RL episode.

## How It Works

**For Beginners:** This contains statistics about one complete episode:

- How much total reward the agent earned
- How many steps it took
- Whether it succeeded (depends on environment definition)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EpisodeResult(Int32,,Int32,Boolean,Boolean,TimeSpan)` | Result of running a single RL episode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Truncated` | Gets whether the episode ended due to reaching max steps (truncated). |

