---
title: "DeploymentConfiguration"
description: "Aggregates all deployment-related configurations."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.Configuration`

Aggregates all deployment-related configurations.
Used to pass deployment settings from AiModelBuilder to AiModelResult.

## Properties

| Property | Summary |
|:-----|:--------|
| `ABTesting` | Gets or sets the A/B testing configuration (null = disabled). |
| `Caching` | Gets or sets the caching configuration (null = use defaults). |
| `Compression` | Gets or sets the compression configuration (null = no compression). |
| `Export` | Gets or sets the export configuration (null = use defaults). |
| `GpuAcceleration` | Gets or sets the GPU acceleration configuration (null = use defaults). |
| `Profiling` | Gets or sets the profiling configuration (null = disabled). |
| `Quantization` | Gets or sets the quantization configuration (null = no quantization). |
| `Telemetry` | Gets or sets the telemetry configuration (null = use defaults). |
| `Versioning` | Gets or sets the versioning configuration (null = use defaults). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(QuantizationConfig,CacheConfig,VersioningConfig,ABTestingConfig,TelemetryConfig,ExportConfig,GpuAccelerationConfig,CompressionConfig,ProfilingConfig)` | Creates a deployment configuration from individual config objects. |

