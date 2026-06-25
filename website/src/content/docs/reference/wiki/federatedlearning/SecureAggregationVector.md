---
title: "SecureAggregationVector<T>"
description: "Implements secure aggregation for vector-based model updates."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Privacy`

Implements secure aggregation for vector-based model updates.

## How It Works

**For Beginners:** Secure aggregation lets the server compute the sum/average of client updates
without learning any single client's update. Each client adds pairwise masks derived from shared secrets.
The masks are constructed so they cancel out in the aggregate.

This implementation provides a synchronous, full-participation secure aggregation mode. If a client
drops out after masks are created, the round must be restarted (dropout-resilient unmasking is a separate
protocol step and is intentionally not part of this in-memory component).

