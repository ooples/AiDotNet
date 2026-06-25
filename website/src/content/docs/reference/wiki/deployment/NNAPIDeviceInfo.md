---
title: "NNAPIDeviceInfo"
description: "Information about an NNAPI-capable device."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Mobile.Android`

Information about an NNAPI-capable device.

## For Beginners

This class describes a hardware accelerator available for NNAPI.
Android devices may have multiple accelerators (CPU, GPU, NPU, DSP) each with different
capabilities and performance characteristics.

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureLevel` | Gets or sets the NNAPI feature level supported by this device. |
| `Name` | Gets or sets the device name (e.g., "Qualcomm Adreno GPU", "Google EdgeTPU"). |
| `PerformanceScore` | Gets or sets the relative performance score (higher is faster, 0-100). |
| `PowerEfficiencyScore` | Gets or sets the relative power efficiency score (higher is more efficient, 0-100). |
| `SupportsFp16` | Gets or sets whether this device supports FP16 operations. |
| `SupportsInt8` | Gets or sets whether this device supports INT8 quantized operations. |
| `Type` | Gets or sets the device type. |

