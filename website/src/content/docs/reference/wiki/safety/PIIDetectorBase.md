---
title: "PIIDetectorBase<T>"
description: "Abstract base class for PII detection modules."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Text`

Abstract base class for PII detection modules.

## For Beginners

This base class provides common code for all PII detectors.
Each detector type extends this class and adds its own way of finding personal
information in text.

## How It Works

Provides shared infrastructure for PII detectors including entity deduplication
and common regex timeout configuration. Concrete implementations provide the actual
detection strategy (regex patterns, NER, context-aware, or composite).

## Methods

| Method | Summary |
|:-----|:--------|
| `DeduplicateEntities(List<PIIEntity>)` | Deduplicates overlapping PII entities, keeping the one with highest confidence. |
| `DetectPII(String)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeoutMs` | Maximum time in milliseconds to allow for regex operations (ReDoS protection). |

