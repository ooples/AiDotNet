---
title: "TFLiteTargetSpec"
description: "TensorFlow Lite target specification with detailed platform requirements."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Mobile.TensorFlowLite`

TensorFlow Lite target specification with detailed platform requirements.

## For Beginners

This class specifies what hardware and software requirements
your TFLite model needs to run. Use this to ensure your model works on your target devices.

## Properties

| Property | Summary |
|:-----|:--------|
| `AndroidMinSdkVersion` | Gets or sets the minimum Android SDK version required (default: 21 = Android 5.0). |
| `Default` | Gets the default specification (CPU-only, wide compatibility). |
| `MinimumIosVersion` | Gets or sets the minimum iOS version for iOS targets (default: "12.0"). |
| `SupportEdgeTpu` | Gets or sets whether to support Edge TPU acceleration (default: false). |
| `SupportGpu` | Gets or sets whether to support GPU acceleration (default: false). |
| `SupportHexagonDsp` | Gets or sets whether to support Hexagon DSP acceleration (default: false). |
| `TargetType` | Gets or sets the target type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForAndroidNnapi` | Creates a spec for Android with NNAPI support. |
| `ForEdgeTpu` | Creates a spec for Google Coral/Edge TPU devices. |
| `ForQualcomm` | Creates a spec for Qualcomm Snapdragon devices. |

