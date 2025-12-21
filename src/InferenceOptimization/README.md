# AiDotNet Inference Optimization

This module provides low-level kernel optimization for critical operations, enabling hardware-specific acceleration for efficient AI model inference.

## Features

### 1. Custom Operator Registration System
- Thread-safe operator registry with automatic fallback
- Priority-based operator selection
- Support for multiple implementations per operation
- Runtime operator switching based on platform capabilities

### 2. Platform Detection
- Automatic detection of CPU architecture (x86/x64, ARM)
- SIMD instruction set detection (SSE, AVX, AVX2, AVX-512, NEON)
- Cache size estimation
- GPU capability detection (CUDA, OpenCL)

### 3. SIMD Vectorization
- AVX2/AVX-512 optimized kernels for x86/x64
- ARM NEON optimized kernels
- Automatic fallback to scalar implementations
- Optimized operations:
  - Vector addition/multiplication
  - Dot product with FMA support
  - ReLU activation
  - Sum reduction
  - Scalar multiply-add

### 4. Optimized Kernels

#### GEMM (General Matrix Multiplication)
- Cache-blocked algorithm for L1 cache efficiency
- Parallel execution for large matrices
- SIMD-optimized inner loops
- Transpose optimization for better memory access patterns
- Expected speedup: 2-3x on AVX2, 2.5x on NEON

#### Fused Attention Kernel
- Scaled dot-product attention: `softmax(QK^T/sqrt(d_k))V`
- Multi-head attention support
- Memory-efficient implementation
- Mask support for causal attention
- Expected speedup: 2.5x

#### Convolution Kernels
- Standard 2D convolution
- Depthwise separable convolution
- Group convolution
- Parallel batch processing
- Expected speedup: 2-2.5x

### 5. CPU Optimizations

#### Cache Optimizer
- L1/L2/L3 cache-aware algorithms
- Automatic tiling parameter computation
- Prefetching for reduced latency
- Cache-aware transpose
- Z-order (Morton) indexing for 2D access patterns
- Cache miss estimation

#### Loop Optimizer
- 2D and 3D loop tiling
- Loop unrolling (4x, 8x)
- Strip mining for cache utilization
- Loop fusion
- Loop interchange optimization
- Parallel tiling with work stealing

### 6. Performance Profiling
- Thread-safe operation tracking
- Timing and memory usage statistics
- Per-operation metrics (min/avg/max/total)
- Performance report generation
- Runtime enable/disable capability

### 7. GPU Optimization Infrastructure
- Base classes for GPU kernel implementations
- Memory management abstractions
- CUDA kernel base (ready for ILGPU/ManagedCuda integration)
- Device capability querying

## Quick Start

```csharp
using AiDotNet.InferenceOptimization;
using AiDotNet.InferenceOptimization.Kernels;
using AiDotNet.Tensors.Engines.Simd;  // SimdKernels location
using AiDotNet.Tensors.LinearAlgebra;

// Initialize the optimization system
OptimizationInitializer.Initialize(enableProfiling: true);

// Use optimized GEMM
var gemmKernel = new GemmKernel();
var a = new Tensor<float>(new[] { 1000, 500 });
var b = new Tensor<float>(new[] { 500, 1000 });
var result = gemmKernel.Execute(a, b);

// Use fused attention
var attentionKernel = new AttentionKernel();
var q = new Tensor<float>(new[] { 1, 128, 64 }); // [batch, seq_len, d_k]
var k = new Tensor<float>(new[] { 1, 128, 64 });
var v = new Tensor<float>(new[] { 1, 128, 64 });
var attended = attentionKernel.Execute(q, k, v);

// Get performance report
var report = OptimizationInitializer.GetPerformanceSummary();
Console.WriteLine(report);
```

## Platform Capabilities

Check what optimizations are available on your platform:

```csharp
var caps = PlatformDetector.Capabilities;
Console.WriteLine($"Best SIMD: {caps.GetBestSimdSet()}");
Console.WriteLine($"Has AVX2: {caps.HasAVX2}");
Console.WriteLine($"Has NEON: {caps.HasNeon}");
Console.WriteLine($"Processor Count: {caps.ProcessorCount}");
```

## Custom Operators

Register your own optimized operators:

```csharp
public class MyCustomKernel : ICustomOperator<float>
{
    public string Name => "MyOperation";
    public string Version => "1.0.0";
    public int Priority => 100;

    public bool IsSupported()
    {
        return PlatformDetector.Capabilities.HasAVX2;
    }

    public double EstimatedSpeedup()
    {
        return 3.0; // Expected 3x speedup
    }

    public Tensor<float> Execute(params Tensor<float>[] inputs)
    {
        // Your optimized implementation
        // ...
    }
}

// Register the operator
CustomOperatorRegistry.Instance.Register(new MyCustomKernel());

// Use the operator
var kernel = CustomOperatorRegistry.Instance.GetOperator<float>("MyOperation");
var result = kernel.Execute(input1, input2);
```

## Performance Profiling

Enable profiling to track performance:

```csharp
// Enable profiling
OptimizationInitializer.Initialize(enableProfiling: true);

// Operations are automatically profiled
// ...

// Get report
var report = OptimizationInitializer.GetPerformanceSummary();
Console.WriteLine(report);

// Reset statistics
OptimizationInitializer.ResetStatistics();
```

## CPU Optimization Utilities

Use cache-aware and loop optimization utilities:

```csharp
using AiDotNet.Tensors.Engines.Optimization;

// Determine optimal tile size
int tileSize = LoopOptimizer.DetermineOptimalTileSize(matrixSize);

// Use tiled loops
LoopOptimizer.Tile2D(rows, cols, tileSize, (iStart, iEnd, jStart, jEnd) =>
{
    // Process tile
});

// Use parallel tiling
LoopOptimizer.ParallelTile2D(rows, cols, tileSize, (iStart, iEnd, jStart, jEnd) =>
{
    // Process tile in parallel
});

// Cache-aware transpose
CacheOptimizer.TransposeBlocked(sourceArray, destArray, rows, cols);
```

## Benchmarking

See `AiDotNetBenchmarkTests/InferenceOptimization/` for benchmark examples.

## Future Enhancements

- GPU kernel implementations using ILGPU or ManagedCuda
- Quantization support (INT8, FP16)
- Model graph optimization
- Operator fusion
- Dynamic batching optimization
- Memory pooling

## Integration with Existing Codebase

The optimization module integrates with existing AiDotNet components:

- **Tensor Operations**: Optimized kernels work with `AiDotNet.LinearAlgebra.Tensor<T>`
- **Neural Networks**: Can be used to accelerate layer operations in `NeuralNetworkBase`
- **Serving**: Integrates with `RequestBatcher` for optimized inference

## Requirements

- .NET 8.0 or later
- x86/x64 or ARM64 processor
- For GPU support: CUDA-capable GPU (future implementation)

## Performance Targets

- 2-5x speedup on critical operations (achieved through SIMD and cache optimization)
- Hardware-specific optimizations (AVX2, AVX-512, NEON)
- Graceful fallback behavior (automatic platform detection)
- Benchmarking against MKL and cuBLAS (future work)

## Contributing

To add new optimizations:

1. Implement `ICustomOperator<T>` interface
2. Override `IsSupported()` to check platform compatibility
3. Implement optimized `Execute()` method
4. Register operator with `CustomOperatorRegistry`
5. Add benchmarks in `AiDotNetBenchmarkTests/`

## License

Same as parent AiDotNet project.

