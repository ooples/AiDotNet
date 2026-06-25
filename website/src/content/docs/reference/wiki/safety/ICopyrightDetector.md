---
title: "ICopyrightDetector<T>"
description: "Interface for copyright and memorization detection modules."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Safety.Text`

Interface for copyright and memorization detection modules.

## For Beginners

A copyright detector checks if an AI's output copies from
copyrighted books, articles, or code. It can detect when the AI is regurgitating
memorized training data rather than generating original content.

## How It Works

Copyright detectors identify potential copyright violations and training data memorization
in model outputs. They use n-gram overlap analysis, embedding similarity to known works,
and perplexity-based memorization detection.

**References:**

- DE-COP: Detecting copyrighted content via paraphrased permutations (2024, arxiv:2402.09910)
- Machine unlearning to remove memorized copyrighted content (2024, arxiv:2412.18621)
- GPTZero: Hierarchical multi-task AI text detection (2026, arxiv:2602.13042)

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMemorizationScore(String)` | Gets the memorization likelihood score (0.0 = original, 1.0 = memorized). |

