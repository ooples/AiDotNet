---
title: "TextAugmentationSettings"
description: "Text-specific augmentation settings with industry-standard defaults."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Augmentation`

Text-specific augmentation settings with industry-standard defaults.

## For Beginners

These settings control how text data is augmented.
Defaults are based on best practices from nlpaug and TextAttack.

## Properties

| Property | Summary |
|:-----|:--------|
| `DeletionRate` | Gets or sets the fraction of words to delete. |
| `EnableBackTranslation` | Gets or sets whether back-translation is enabled. |
| `EnableRandomDeletion` | Gets or sets whether random deletion is enabled. |
| `EnableRandomInsertion` | Gets or sets whether random insertion is enabled. |
| `EnableRandomSwap` | Gets or sets whether random swap is enabled. |
| `EnableSynonymReplacement` | Gets or sets whether synonym replacement is enabled. |
| `InsertionRate` | Gets or sets the fraction of words to insert. |
| `NumSwaps` | Gets or sets the number of word swaps to perform. |
| `SynonymReplacementRate` | Gets or sets the fraction of words to replace with synonyms. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetConfiguration` | Gets the configuration as a dictionary. |

