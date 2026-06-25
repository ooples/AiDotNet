---
title: "PubLayNetDataLoader<T>"
description: "Loads the PubLayNet document layout analysis dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the PubLayNet document layout analysis dataset.

## How It Works

PubLayNet expects:

Features are flattened image pixels Tensor[N, H * W * 3].
Labels are layout region count per class Tensor[N, NumClasses].

