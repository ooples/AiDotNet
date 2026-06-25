---
title: "SCNet<T>"
description: "SCNet (Sparse Compression Network) for music source separation (Tong et al., 2024)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.SourceSeparation`

SCNet (Sparse Compression Network) for music source separation (Tong et al., 2024).

## For Beginners

Instead of processing every individual frequency separately (which is slow),
SCNet groups similar frequencies into clusters, processes the clusters efficiently, then expands
back to full detail. Think of it like editing a compressed photo—you can make changes much faster,
and the results still look great when decompressed.

**Usage:**

## How It Works

SCNet uses sparse compression to reduce the frequency dimension before processing with
attention layers, achieving competitive separation quality with significantly fewer parameters.
It compresses frequency features into compact cluster representations, processes them, then
decompresses back to the full frequency resolution for mask estimation.

