---
title: "DocumentFrequencyFS<T>"
description: "Document Frequency based feature selection for text data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.NLP`

Document Frequency based feature selection for text data.

## For Beginners

Some words appear in almost every document (like "is" or "the")
and don't help distinguish documents. Other words appear only once or twice in the
entire corpus and might be typos or too specific. DF filtering keeps words that appear
in a reasonable number of documents - common enough to be meaningful but not so common
that everyone uses them.

## How It Works

Document Frequency (DF) selection filters terms based on how many documents
they appear in. Terms appearing in too few or too many documents are removed,
as they are either too rare to be useful or too common to be discriminative.

