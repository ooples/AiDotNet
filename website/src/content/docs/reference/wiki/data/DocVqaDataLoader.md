---
title: "DocVqaDataLoader<T>"
description: "Loads the DocVQA document visual question answering dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the DocVQA document visual question answering dataset.

## How It Works

DocVQA expects:

Features are flattened image pixels Tensor[N, H * W * 3].
Labels are answer text encoded as character indices Tensor[N, MaxAnswerLength],
where each element is the Unicode code point of the character.

