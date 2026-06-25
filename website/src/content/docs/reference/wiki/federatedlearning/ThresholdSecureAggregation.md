---
title: "ThresholdSecureAggregation<T>"
description: "Implements dropout-resilient secure aggregation for structured (layered) model updates."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Privacy`

Implements dropout-resilient secure aggregation for structured (layered) model updates.

## How It Works

**For Beginners:** Some models expose parameters as a dictionary of named arrays (for example, one array per layer).
This wrapper adapts that representation to the vector-based secure aggregation core.

