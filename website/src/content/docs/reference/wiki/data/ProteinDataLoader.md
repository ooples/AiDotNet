---
title: "ProteinDataLoader<T>"
description: "Loads protein structure datasets as flattened feature/label tensors for graph-level classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Graph`

Loads protein structure datasets as flattened feature/label tensors for graph-level classification.

## How It Works

Expects a directory structure with protein files in CSV or TSV format:

Features are flattened residue features Tensor[N, MaxResidues * FeatureDimension].
Labels are functional class index Tensor[N, 1].

