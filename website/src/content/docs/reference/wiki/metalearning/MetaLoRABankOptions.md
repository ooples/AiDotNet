---
title: "MetaLoRABankOptions<T, TInput, TOutput>"
description: "Configuration options for Meta-LoRA Bank (2024)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Meta-LoRA Bank (2024).

## How It Works

Meta-LoRA Bank maintains a bank of diverse LoRA modules. For a new task, the algorithm
selects and combines the most relevant modules using a task-conditioned gating mechanism.
Meta-learning optimizes both the LoRA modules in the bank and the gating network.

## Properties

| Property | Summary |
|:-----|:--------|
| `BankSize` | Number of LoRA modules in the bank. |
| `GatingTemperature` | Softmax temperature for module selection gating. |
| `LoadBalanceRegularization` | Load balancing regularization to encourage uniform module utilization. |
| `Rank` | Rank of each individual LoRA module. |
| `TopK` | Number of top modules to select per task (top-K gating). |

