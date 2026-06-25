---
title: "DeploymentRuntime<T>"
description: "Runtime environment for deployed models with warm-up, versioning, A/B testing, and telemetry."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Runtime`

Runtime environment for deployed models with warm-up, versioning, A/B testing, and telemetry.

## For Beginners

DeploymentRuntime provides AI safety functionality. Default values follow the original paper settings.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetModelStatistics(String,String)` | Gets telemetry statistics for a model. |
| `GetRegisteredModels` | Gets all registered model versions. |
| `InferAsync(String,String,[])` | Performs inference with the specified model version. |
| `InferWithABTestAsync(String,[])` | Performs inference with A/B testing (automatically selects version based on traffic split). |
| `RegisterModel(String,String,String,Dictionary<String,Object>)` | Registers a model version with the runtime. |
| `SetupABTest(String,String,String,String,Double)` | Sets up A/B testing between two model versions. |
| `WarmUpModelAsync(String,String,Int32)` | Warms up a model by running inference on dummy data. |

