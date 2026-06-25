---
title: "TfIdfFS<T>"
description: "TF-IDF based feature selection for text data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.NLP`

TF-IDF based feature selection for text data.

## For Beginners

In text data, some words appear often but are
meaningless (like "the"), while others are rare but important. TF-IDF balances
term frequency (how often a word appears in a document) with inverse document
frequency (how rare it is across all documents). Words that are common in some
documents but rare overall get high scores.

## How It Works

TF-IDF (Term Frequency-Inverse Document Frequency) based selection identifies
features (terms) that are both frequent within documents and discriminative
across the corpus. Features with high TF-IDF scores are informative for
distinguishing documents.

