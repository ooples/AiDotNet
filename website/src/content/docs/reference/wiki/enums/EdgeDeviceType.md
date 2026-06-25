---
title: "EdgeDeviceType"
description: "Types of edge devices for optimization targeting."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Types of edge devices for optimization targeting.

## How It Works

**For Beginners:** Edge devices are small computers that run AI models locally instead of
in the cloud. Each type of edge device has different hardware capabilities, so the library
can optimize your model specifically for the device you're targeting:

- **Generic**: Works on any edge device, but not specifically optimized
- **RaspberryPi**: Popular single-board computer for hobbyists and education
- **Jetson**: NVIDIA's edge AI platform with GPU acceleration
- **CoralTPU**: Google's Edge TPU for fast AI inference
- **Movidius**: Intel's vision processing unit for cameras and drones
- **Microcontroller**: Very small devices like Arduino or ESP32 with minimal resources
- **AndroidPhone**: Android smartphones and tablets
- **iOS**: iPhones and iPads

Choosing the right device type helps the library apply device-specific optimizations.

## Fields

| Field | Summary |
|:-----|:--------|
| `AndroidPhone` | Android phone - smartphone with potential GPU/NPU acceleration |
| `CoralTPU` | Google Coral Edge TPU - specialized AI accelerator ($25-$150) |
| `Generic` | Generic edge device - works everywhere but not specifically optimized |
| `Jetson` | NVIDIA Jetson - edge AI platform with GPU acceleration ($99-$500) |
| `Microcontroller` | Microcontroller (STM32, ESP32, etc.) - very limited resources |
| `Movidius` | Intel Movidius - vision processing unit for cameras and drones |
| `RaspberryPi` | Raspberry Pi - popular single-board computer ($35-$100) |
| `iOS` | iPhone/iPad - iOS devices with Neural Engine |

