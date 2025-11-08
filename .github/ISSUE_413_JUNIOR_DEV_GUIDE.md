# Junior Developer Implementation Guide: Issue #413

## Overview
**Issue**: Custom Kernel Optimization
**Goal**: Implement SIMD-optimized kernels for critical operations
**Difficulty**: Expert
**Estimated Time**: 20-24 hours

## What are Custom Kernels?

Custom kernels are highly-optimized implementations of mathematical operations:
- **SIMD (Single Instruction Multiple Data)**: Process multiple values in parallel
- **Vectorization**: Use CPU vector instructions (SSE, AVX, AVX-512)
- **Cache Optimization**: Minimize cache misses through data locality
- **Loop Unrolling**: Reduce loop overhead

## SIMD in C#

C# provides `System.Numerics.Vector<T>` for SIMD operations:

```csharp
using System.Numerics;

// Traditional scalar code
for (int i = 0; i < array.Length; i++)
{
    result[i] = array[i] * 2.0f;
}

// SIMD vectorized code
int vectorSize = Vector<float>.Count; // Typically 4, 8, or 16
for (int i = 0; i < array.Length; i += vectorSize)
{
    var vector = new Vector<float>(array, i);
    var doubled = vector * 2.0f;
    doubled.CopyTo(result, i);
}
```

## Key Operations to Optimize

### 1. Matrix Multiplication (GEMM)

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Kernels\OptimizedMatrixOps.cs
using System.Numerics;

namespace AiDotNet.Kernels
{
    public static class OptimizedMatrixOps
    {
        /// <summary>
        /// SIMD-optimized matrix multiplication C = A * B.
        /// Uses cache blocking and vectorization.
        /// </summary>
        public static unsafe void MatMul(
            float* a, int aRows, int aCols,
            float* b, int bRows, int bCols,
            float* c)
        {
            const int blockSize = 64; // Cache-friendly block size

            for (int i = 0; i < aRows; i++)
            {
                for (int j = 0; j < bCols; j++)
                {
                    // Initialize accumulator
                    float sum = 0.0f;

                    // Vectorized inner product
                    int k = 0;
                    int vectorSize = Vector<float>.Count;

                    // Process in SIMD chunks
                    for (; k <= aCols - vectorSize; k += vectorSize)
                    {
                        var vecA = new Vector<float>(a + i * aCols + k);
                        var vecB = LoadColumn(b, bCols, k, j, vectorSize);

                        var product = vecA * vecB;
                        sum += Vector.Dot(vecA, vecB);
                    }

                    // Handle remainder
                    for (; k < aCols; k++)
                    {
                        sum += a[i * aCols + k] * b[k * bCols + j];
                    }

                    c[i * bCols + j] = sum;
                }
            }
        }

        private static unsafe Vector<float> LoadColumn(
            float* matrix, int cols, int row, int col, int count)
        {
            Span<float> temp = stackalloc float[Vector<float>.Count];

            for (int i = 0; i < count; i++)
            {
                temp[i] = matrix[(row + i) * cols + col];
            }

            return new Vector<float>(temp);
        }

        /// <summary>
        /// Cache-blocked matrix multiplication (tiled).
        /// Better cache locality for large matrices.
        /// </summary>
        public static unsafe void MatMulBlocked(
            float* a, int aRows, int aCols,
            float* b, int bRows, int bCols,
            float* c, int blockSize = 64)
        {
            // Zero initialize C
            for (int i = 0; i < aRows * bCols; i++)
                c[i] = 0.0f;

            // Blocked multiplication
            for (int i0 = 0; i0 < aRows; i0 += blockSize)
            {
                for (int j0 = 0; j0 < bCols; j0 += blockSize)
                {
                    for (int k0 = 0; k0 < aCols; k0 += blockSize)
                    {
                        // Process block
                        int iMax = Math.Min(i0 + blockSize, aRows);
                        int jMax = Math.Min(j0 + blockSize, bCols);
                        int kMax = Math.Min(k0 + blockSize, aCols);

                        for (int i = i0; i < iMax; i++)
                        {
                            for (int k = k0; k < kMax; k++)
                            {
                                float aVal = a[i * aCols + k];

                                for (int j = j0; j < jMax; j++)
                                {
                                    c[i * bCols + j] += aVal * b[k * bCols + j];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
```

### 2. Convolution

```csharp
/// <summary>
/// SIMD-optimized 2D convolution.
/// Uses im2col transformation for GEMM-based conv.
/// </summary>
public static unsafe void Conv2D(
    float* input, int batchSize, int inChannels, int height, int width,
    float* filters, int outChannels, int kernelSize,
    float* output, int stride = 1, int padding = 0)
{
    int outHeight = (height + 2 * padding - kernelSize) / stride + 1;
    int outWidth = (width + 2 * padding - kernelSize) / stride + 1;

    // Im2col: unfold input into matrix for GEMM
    int colHeight = kernelSize * kernelSize * inChannels;
    int colWidth = outHeight * outWidth;

    float* colBuffer = stackalloc float[colHeight * colWidth];

    Im2Col(input, height, width, inChannels, kernelSize, stride, padding, colBuffer);

    // Reshape filters to matrix [outChannels, colHeight]
    // Perform GEMM: output = filters * colBuffer
    MatMul(filters, outChannels, colHeight, colBuffer, colHeight, colWidth, output);
}

private static unsafe void Im2Col(
    float* input, int height, int width, int channels,
    int kernelSize, int stride, int padding,
    float* colBuffer)
{
    int outHeight = (height + 2 * padding - kernelSize) / stride + 1;
    int outWidth = (width + 2 * padding - kernelSize) / stride + 1;

    int colIdx = 0;

    for (int c = 0; c < channels; c++)
    {
        for (int ky = 0; ky < kernelSize; ky++)
        {
            for (int kx = 0; kx < kernelSize; kx++)
            {
                for (int y = 0; y < outHeight; y++)
                {
                    for (int x = 0; x < outWidth; x++)
                    {
                        int inputY = y * stride + ky - padding;
                        int inputX = x * stride + kx - padding;

                        if (inputY >= 0 && inputY < height && inputX >= 0 && inputX < width)
                        {
                            colBuffer[colIdx++] = input[c * height * width + inputY * width + inputX];
                        }
                        else
                        {
                            colBuffer[colIdx++] = 0.0f; // Padding
                        }
                    }
                }
            }
        }
    }
}
```

### 3. Activation Functions

```csharp
/// <summary>
/// SIMD-optimized ReLU: max(0, x)
/// </summary>
public static void ReLU_SIMD(float[] input, float[] output)
{
    var zero = Vector<float>.Zero;
    int vectorSize = Vector<float>.Count;

    int i = 0;
    for (; i <= input.Length - vectorSize; i += vectorSize)
    {
        var vec = new Vector<float>(input, i);
        var result = Vector.Max(vec, zero);
        result.CopyTo(output, i);
    }

    // Handle remainder
    for (; i < input.Length; i++)
    {
        output[i] = Math.Max(0, input[i]);
    }
}

/// <summary>
/// SIMD-optimized element-wise addition.
/// </summary>
public static void Add_SIMD(float[] a, float[] b, float[] result)
{
    int vectorSize = Vector<float>.Count;

    int i = 0;
    for (; i <= a.Length - vectorSize; i += vectorSize)
    {
        var vecA = new Vector<float>(a, i);
        var vecB = new Vector<float>(b, i);
        var sum = vecA + vecB;
        sum.CopyTo(result, i);
    }

    for (; i < a.Length; i++)
    {
        result[i] = a[i] + b[i];
    }
}
```

### 4. Softmax Optimization

```csharp
/// <summary>
/// Optimized softmax with numerical stability.
/// </summary>
public static void Softmax(float[] logits, float[] output)
{
    // Find max for numerical stability
    float max = float.MinValue;
    for (int i = 0; i < logits.Length; i++)
    {
        if (logits[i] > max)
            max = logits[i];
    }

    // Compute exp(x - max) and sum
    float sum = 0.0f;

    int vectorSize = Vector<float>.Count;
    var maxVec = new Vector<float>(max);
    var sumVec = Vector<float>.Zero;

    int i = 0;
    for (; i <= logits.Length - vectorSize; i += vectorSize)
    {
        var vec = new Vector<float>(logits, i);
        var shifted = vec - maxVec;

        // Approximate exp with Taylor series or use scalar for accuracy
        Span<float> temp = stackalloc float[vectorSize];
        shifted.CopyTo(temp);

        for (int j = 0; j < vectorSize; j++)
        {
            temp[j] = MathF.Exp(temp[j]);
            sum += temp[j];
        }

        var expVec = new Vector<float>(temp);
        expVec.CopyTo(output, i);
    }

    // Remainder
    for (; i < logits.Length; i++)
    {
        output[i] = MathF.Exp(logits[i] - max);
        sum += output[i];
    }

    // Normalize
    var sumVecFinal = new Vector<float>(sum);

    i = 0;
    for (; i <= logits.Length - vectorSize; i += vectorSize)
    {
        var vec = new Vector<float>(output, i);
        var normalized = vec / sumVecFinal;
        normalized.CopyTo(output, i);
    }

    for (; i < logits.Length; i++)
    {
        output[i] /= sum;
    }
}
```

## Benchmarking

```csharp
using BenchmarkDotNet.Attributes;

[MemoryDiagnoser]
public class MatMulBenchmark
{
    private float[] _a, _b, _c;

    [GlobalSetup]
    public void Setup()
    {
        int n = 1024;
        _a = new float[n * n];
        _b = new float[n * n];
        _c = new float[n * n];

        var rand = new Random(42);
        for (int i = 0; i < n * n; i++)
        {
            _a[i] = (float)rand.NextDouble();
            _b[i] = (float)rand.NextDouble();
        }
    }

    [Benchmark(Baseline = true)]
    public void MatMul_Naive()
    {
        // Naive O(n³) implementation
        int n = 1024;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += _a[i * n + k] * _b[k * n + j];
                }
                _c[i * n + j] = sum;
            }
        }
    }

    [Benchmark]
    public unsafe void MatMul_SIMD()
    {
        fixed (float* a = _a, b = _b, c = _c)
        {
            OptimizedMatrixOps.MatMul(a, 1024, 1024, b, 1024, 1024, c);
        }
    }

    [Benchmark]
    public unsafe void MatMul_Blocked()
    {
        fixed (float* a = _a, b = _b, c = _c)
        {
            OptimizedMatrixOps.MatMulBlocked(a, 1024, 1024, b, 1024, 1024, c);
        }
    }
}
```

## Performance Tips

### 1. Memory Alignment

```csharp
// Allocate aligned memory for SIMD
var aligned = new float[1024 + Vector<float>.Count];
int offset = GetAlignmentOffset(aligned);
Span<float> alignedSpan = aligned.AsSpan(offset, 1024);
```

### 2. Cache Optimization

**Cache Line Size**: 64 bytes (16 floats)
**L1 Cache**: ~32 KB
**L2 Cache**: ~256 KB
**L3 Cache**: ~8 MB

**Strategy**: Keep working set in L1/L2 cache through blocking.

### 3. Loop Unrolling

```csharp
// Manual unrolling (4x)
for (int i = 0; i < n; i += 4)
{
    result[i] = input[i] * 2;
    result[i + 1] = input[i + 1] * 2;
    result[i + 2] = input[i + 2] * 2;
    result[i + 3] = input[i + 3] * 2;
}
```

## Platform-Specific Optimizations

### AVX-512 (when available)

```csharp
if (Avx512F.IsSupported)
{
    // Use 512-bit vectors (16 floats at once)
    Vector512<float> vec512 = Vector512.Load(ptr);
    var result = Vector512.Multiply(vec512, scalar);
}
```

### ARM NEON

```csharp
if (AdvSimd.IsSupported)
{
    // ARM NEON instructions
    Vector128<float> vec = AdvSimd.LoadVector128(ptr);
    var result = AdvSimd.Multiply(vec, scalar);
}
```

## Testing

```csharp
[Fact]
public void MatMul_SIMD_MatchesNaive()
{
    float[] a = { 1, 2, 3, 4 };
    float[] b = { 5, 6, 7, 8 };
    float[] c_naive = new float[4];
    float[] c_simd = new float[4];

    // Naive
    MatMulNaive(a, 2, 2, b, 2, 2, c_naive);

    // SIMD
    unsafe
    {
        fixed (float* pa = a, pb = b, pc = c_simd)
        {
            OptimizedMatrixOps.MatMul(pa, 2, 2, pb, 2, 2, pc);
        }
    }

    // Results should match
    for (int i = 0; i < 4; i++)
    {
        Assert.Equal(c_naive[i], c_simd[i], precision: 5);
    }
}
```

## Learning Resources

- **SIMD Programming**: https://learn.microsoft.com/en-us/dotnet/standard/simd
- **Cache Optimization**: https://www.intel.com/content/www/us/en/developer/articles/technical/cache-blocking-techniques.html
- **BenchmarkDotNet**: https://benchmarkdotnet.org/

---

**Good luck!** Custom kernel optimization can provide 5-10x speedups for critical operations. Use benchmarking to validate improvements!
