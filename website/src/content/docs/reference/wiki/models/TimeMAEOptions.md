---
title: "TimeMAEOptions<T>"
description: "Configuration options for TimeMAE (Masked Autoencoder for Time Series)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TimeMAE (Masked Autoencoder for Time Series).

## How It Works

TimeMAE applies masked autoencoding to time series, randomly masking patches of the input
and training a transformer to reconstruct the missing patches, learning rich temporal representations.

**Reference:** Cheng et al., "TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders", 2023.

