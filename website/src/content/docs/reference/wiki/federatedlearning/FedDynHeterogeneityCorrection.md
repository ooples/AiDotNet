---
title: "FedDynHeterogeneityCorrection<T>"
description: "FedDyn-style dynamic regularization using a per-client drift accumulator."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Heterogeneity`

FedDyn-style dynamic regularization using a per-client drift accumulator.

## How It Works

**For Beginners:** FedDyn reduces client drift by maintaining an extra per-client state that
accumulates how the client tends to move away from the global model.

