---
title: "RuntimeConfiguration"
description: "Configuration for the deployment runtime environment."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.Runtime`

Configuration for the deployment runtime environment.

## For Beginners

RuntimeConfiguration provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoWarmUp` | Gets or sets whether to enable automatic model warm-up on registration (default: true). |
| `CacheEvictionPolicy` | Gets or sets the cache eviction policy. |
| `CacheSizeMB` | Gets or sets the cache size in megabytes (default: 100 MB). |
| `EnableABTesting` | Gets or sets whether to enable A/B testing support (default: false). |
| `EnableCaching` | Gets or sets whether to enable model caching (default: true). |
| `EnableGpuAcceleration` | Gets or sets whether to enable GPU acceleration for inference (default: true). |
| `EnableHealthChecks` | Gets or sets whether to enable health checks (default: true). |
| `EnablePerformanceMonitoring` | Gets or sets whether to enable performance monitoring (default: true). |
| `EnableTelemetry` | Gets or sets whether to enable telemetry collection (default: true). |
| `EnableVersioning` | Gets or sets whether to enable versioning (default: true). |
| `HealthCheckIntervalSeconds` | Gets or sets the health check interval in seconds (default: 300 = 5 minutes). |
| `InferenceTimeoutSeconds` | Gets or sets the inference timeout in seconds (default: 30). |
| `MaxVersionsPerModel` | Gets or sets the maximum number of versions to keep per model (default: 3). |
| `ModelLoadTimeoutSeconds` | Gets or sets the model load timeout in seconds (default: 60). |
| `PerformanceAlertThresholdMs` | Gets or sets the performance alert threshold in milliseconds (default: 1000). |
| `TelemetryBufferSize` | Gets or sets the telemetry buffer size (default: 1000 events). |
| `TelemetryFlushIntervalSeconds` | Gets or sets the telemetry flush interval in seconds (default: 60). |
| `TelemetrySamplingRate` | Gets or sets the telemetry sampling rate (0.0 to 1.0, default: 1.0 = 100%). |
| `WarmUpIterations` | Gets or sets the number of warm-up iterations (default: 10). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForDevelopment` | Creates a configuration for development/testing. |
| `ForEdge` | Creates a minimal configuration for edge devices. |
| `ForProduction` | Creates a configuration for production deployment. |

