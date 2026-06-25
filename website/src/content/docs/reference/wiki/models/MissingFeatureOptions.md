---
title: "MissingFeatureOptions"
description: "Configuration for handling missing features in vertical federated learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration for handling missing features in vertical federated learning.

## For Beginners

In real-world VFL deployments, not all parties have data for
all entities. For example, a bank might have records for 100,000 customers, but the
partnering hospital only has records for 30,000 of those patients. The other 70,000
customers have "missing" hospital features.

## How It Works

This class controls how those missing features are handled during training
and inference.

Example:

## Properties

| Property | Summary |
|:-----|:--------|
| `AddMissingnessIndicator` | Gets or sets whether to create a binary indicator feature marking which features are imputed vs. |
| `AlignmentThreshold` | Gets or sets the alignment confidence threshold when using fuzzy entity matching. |
| `AllowPartialAlignment` | Gets or sets whether to include partially-aligned entities in training. |
| `MinimumOverlapRatio` | Gets or sets the minimum required overlap ratio between parties before training can proceed. |
| `Strategy` | Gets or sets the strategy for imputing missing features. |

