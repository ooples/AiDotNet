---
layout: default
title: Getting Started
nav_order: 2
has_children: true
permalink: /getting-started/
---

# Getting Started with AiDotNet
{: .no_toc }

This guide will help you install AiDotNet and build your first AI model.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Prerequisites

Before installing AiDotNet, ensure you have:

- **.NET 8.0 SDK** or later ([Download](https://dotnet.microsoft.com/download))
- (Optional) **NVIDIA GPU** with CUDA for GPU acceleration

## Installation

### NuGet Package Manager

```bash
dotnet add package AiDotNet
```

### Package Manager Console (Visual Studio)

```powershell
Install-Package AiDotNet
```

### PackageReference (csproj)

```xml
<PackageReference Include="AiDotNet" Version="*" />
```

## Your First Model

Let's build a simple classification model:

```csharp
using AiDotNet;
using AiDotNet.Classification;

// Sample data
var features = new double[][]
{
    new[] { 5.1, 3.5, 1.4, 0.2 },
    new[] { 7.0, 3.2, 4.7, 1.4 },
    new[] { 6.3, 3.3, 6.0, 2.5 }
};
var labels = new double[] { 0, 1, 2 };

// Build and train
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new RandomForestClassifier<double>(nEstimators: 100))
    .ConfigurePreprocessing()  // Auto-applies StandardScaler
    .BuildAsync(features, labels);

// Predict using the result object (facade pattern)
var prediction = result.Predict(new[] { 5.9, 3.0, 5.1, 1.8 });
Console.WriteLine($"Predicted class: {prediction}");
```

## Core Concepts

### AiModelBuilder

The `AiModelBuilder` is the main entry point for building models. It uses a fluent API:

```csharp
var builder = new AiModelBuilder<T, TInput, TOutput>();
```

Where:
- `T` - Numeric type (usually `double` or `float`)
- `TInput` - Input data type (e.g., `double[]`, `Tensor<T>`)
- `TOutput` - Output type (e.g., `double`, `int`)

### Configuration Methods

The builder provides many `Configure*` methods:

| Method | Purpose |
|:-------|:--------|
| `ConfigureModel()` | Set the ML model/algorithm |
| `ConfigureOptimizer()` | Set the optimizer (for neural networks) |
| `ConfigurePreprocessing()` | Add data preprocessing steps |
| `ConfigurePostprocessing()` | Add output postprocessing |
| `ConfigureCrossValidation()` | Enable cross-validation |
| `ConfigureGpuAcceleration()` | Enable GPU training |

### Building and Training

Call `BuildAsync()` with your training data:

```csharp
var result = await builder.BuildAsync(features, labels);
```

The `result` contains:
- `Model` - The trained model
- `CrossValidationResult` - CV metrics (if enabled)
- `TrainingMetrics` - Training statistics

## Next Steps

- [Installation Guide](./installation) - Detailed installation instructions
- [Quick Start Tutorial](./quickstart) - Build your first model
- [Core Concepts](./concepts) - Understand the architecture
- [Samples](../samples/) - Browse complete examples

## GPU Acceleration

To enable GPU training:

```csharp
builder.ConfigureGpuAcceleration(new GpuAccelerationConfig
{
    Enabled = true,
    DeviceId = 0  // Use first GPU
});
```

Requirements:
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8+
- cuDNN 8.6+

## Need Help?

- [FAQ](./faq) - Frequently asked questions
- [Troubleshooting](./troubleshooting) - Common issues and solutions
- [GitHub Issues](https://github.com/ooples/AiDotNet/issues) - Report bugs
- [GitHub Discussions](https://github.com/ooples/AiDotNet/discussions) - Ask questions
