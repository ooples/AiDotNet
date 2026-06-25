---
title: "StreamingTextDataset<T>"
description: "A streaming text dataset that lazily reads and tokenizes text files for language model training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text`

A streaming text dataset that lazily reads and tokenizes text files for language model training.

## How It Works

Reads text files sequentially, splits into fixed-length token sequences using a simple
word-level tokenizer. Input is tokens[0..n-1], target is tokens[1..n] (next-token prediction).
Suitable for pre-training language models on large text corpora.

