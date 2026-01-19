---
layout: default
title: Installation
parent: Getting Started
nav_order: 1
---

# Installation Guide
{: .no_toc }

Complete guide to installing AiDotNet and its dependencies.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Basic Installation

### Using .NET CLI

```bash
dotnet add package AiDotNet
```

### Using Package Manager Console

```powershell
Install-Package AiDotNet
```

### Using PackageReference

Add to your `.csproj` file:

```xml
<ItemGroup>
  <PackageReference Include="AiDotNet" Version="*" />
</ItemGroup>
```

## Platform Support

| Platform | Status | Notes |
|:---------|:-------|:------|
| Windows x64 | ✅ Full | Recommended |
| Linux x64 | ✅ Full | Ubuntu 20.04+ |
| macOS x64 | ✅ Full | macOS 11+ |
| macOS ARM64 | ✅ Full | Apple Silicon |

## .NET Version Support

| Version | Status |
|:--------|:-------|
| .NET 8.0 | ✅ Primary target |
| .NET 7.0 | ✅ Supported |
| .NET 6.0 | ✅ Supported |
| .NET Framework 4.6.2+ | ✅ Supported |

## GPU Acceleration

### NVIDIA CUDA

For GPU-accelerated training on NVIDIA GPUs:

1. **Install CUDA Toolkit 12.x+ (or 11.8+ minimum)**
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - CUDA 12.x recommended for best performance with modern GPUs

2. **Install cuDNN 9.x+ (or 8.6+ minimum)**
   - Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - cuDNN 9.x recommended for compatibility with CUDA 12.x

3. **Verify installation**
   ```bash
   nvcc --version
   ```

### OpenCL

For AMD/Intel GPU support:

1. **Install OpenCL runtime** for your GPU vendor
2. **Install CLBlast** for optimized BLAS operations

## Optional Packages

### Model Serving

For production model serving:

```bash
dotnet add package AiDotNet.Serving
```

### Dashboard

For training visualization:

```bash
dotnet add package AiDotNet.Dashboard
```

## Verifying Installation

Create a simple test:

```csharp
using AiDotNet;

Console.WriteLine("AiDotNet installed successfully!");

// Test basic functionality
var builder = new AiModelBuilder<double, double[], double>();
Console.WriteLine("AiModelBuilder created.");
```

Run it:

```bash
dotnet run
```

## Troubleshooting

### Package Not Found

Ensure you have the latest NuGet sources:

```bash
dotnet nuget list source
```

### GPU Not Detected

1. Verify CUDA installation: `nvcc --version`
2. Check GPU visibility: `nvidia-smi`
3. Ensure CUDA path is in environment variables

### Build Errors

1. Clear NuGet cache: `dotnet nuget locals all --clear`
2. Restore packages: `dotnet restore`
3. Rebuild: `dotnet build`

## Next Steps

- [Quick Start Tutorial](./quickstart) - Build your first model
- [GPU Setup Guide](./gpu-setup) - Detailed GPU configuration
