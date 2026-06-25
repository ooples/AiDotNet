---
title: "TextFeatureSelector<T>"
description: "Feature selection for text/NLP data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.DomainSpecific`

Feature selection for text/NLP data.

## For Beginners

Text data has unique properties - some words are
common but uninformative (like "the"), while rare words might be very
distinctive. This selector picks words/features that best distinguish
between different text categories.

## How It Works

TextFeatureSelector is designed for selecting features from text data,
such as word frequencies, TF-IDF scores, or word embeddings. It considers
text-specific properties like term frequency and document frequency.

