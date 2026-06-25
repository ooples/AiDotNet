---
title: "ExFMScheduler<T>"
description: "ExFM (Exponential Flow Matching) scheduler with exponential time discretization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

ExFM (Exponential Flow Matching) scheduler with exponential time discretization.

## For Beginners

ExFM places more computation effort on the final "polishing" steps
where fine details are added, and less on the early steps where just the rough shape
forms. This produces sharper images with the same number of total steps.

## How It Works

Uses exponential (log-space) time discretization instead of uniform spacing for flow
matching. This concentrates more steps near t=0 (clean data) where fine details matter
and fewer steps near t=1 (noise) where coarse structure is determined.

## Methods

| Method | Summary |
|:-----|:--------|
| `SetTimesteps(Int32)` |  |
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` |  |

