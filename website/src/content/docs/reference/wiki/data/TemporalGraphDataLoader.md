---
title: "TemporalGraphDataLoader<T>"
description: "Loads temporal graph datasets (timestamped interactions for dynamic link prediction)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Graph`

Loads temporal graph datasets (timestamped interactions for dynamic link prediction).

## How It Works

Expects CSV/TSV interaction files:

Features are interaction features Tensor[N, NodeFeatureDim + EdgeFeatureDim].
Labels are interaction label Tensor[N, 1].

