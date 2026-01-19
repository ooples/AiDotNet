using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// High-performance convolution using im2col fusion and cache tiling.
/// Eliminates explicit im2col buffer by computing indices on-the-fly during GEMM.
/// </summary>
internal static class FusedConvHelper
{
    // Cache tile sizes optimized for typical L1 cache (32KB)
    // Each tile processes McxNc output elements with Kc input channels
    private const int Mc = 64;  // Output channels tile
    private const int Nc = 256; // Spatial positions tile
    private const int Kc = 64;  // Input channels * kernel elements tile

    // Micro-kernel sizes for register blocking
    private const int Mr = 8;  // Rows processed by micro-kernel
    private const int Nr = 8;  // Columns processed by micro-kernel

    private static readonly bool UseAvx2 = Avx2.IsSupported;
    private static readonly bool UseFma = Fma.IsSupported;

    /// <summary>
    /// Check if fused convolution should be used for this configuration.
    /// </summary>
    public static bool ShouldUseFusedConv(int kernelH, int kernelW, int strideH, int strideW,
        int outputH, int outputW, int inChannels, int outChannels)
    {
        if (!UseAvx2) return false;

        // Use fused conv for medium-to-large convolutions where cache efficiency matters
        int outputSize = outputH * outputW;
        int kernelElements = kernelH * kernelW * inChannels;

        // Fused conv is beneficial when:
        // 1. Output is large enough to benefit from tiling
        // 2. Kernel is not too small (overhead of index computation)
        return outputSize >= 1024 && kernelElements >= 9;
    }

    /// <summary>
    /// Performs Conv2D using fused im2col-GEMM with cache tiling.
    /// </summary>
    public static unsafe void Conv2DFused(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> kernel,
        Span<float> output,
        int batch, int inChannels, int height, int width,
        int outChannels, int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW, int outHeight, int outWidth)
    {
        int outputSize = outHeight * outWidth;
        int kernelSize = kernelH * kernelW;
        int K = inChannels * kernelSize; // Total columns in im2col (row dimension for kernel)

        // Clear output
        output.Clear();

        fixed (float* inputPtr = input)
        fixed (float* kernelPtr = kernel)
        fixed (float* outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
            {
                float* inputBatch = inputPtr + b * inChannels * height * width;
                float* outputBatch = outputPtr + b * outChannels * outputSize;

                // Tiled GEMM with fused im2col
                // C[M,N] = A[M,K] @ B[K,N] where A=kernel, B=im2col(input), C=output
                TiledGemmFusedIm2Col(
                    kernelPtr, inputBatch, outputBatch,
                    outChannels, outputSize, K,
                    height, width, outHeight, outWidth,
                    kernelH, kernelW, strideH, strideW,
                    padH, padW, dilationH, dilationW, inChannels);
            }
        }
    }

    /// <summary>
    /// Tiled GEMM with fused im2col computation.
    /// Uses cache blocking for optimal memory access patterns.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void TiledGemmFusedIm2Col(
        float* A,      // Kernel: [outChannels, K]
        float* input,  // Input: [inChannels, height, width]
        float* C,      // Output: [outChannels, outputSize]
        int M,         // outChannels
        int N,         // outputSize
        int K,         // inChannels * kernelH * kernelW
        int height, int width, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW,
        int padH, int padW, int dilationH, int dilationW, int inChannels)
    {
        int kernelSize = kernelH * kernelW;

        // Parallel over output channel tiles for large convolutions
        bool useParallel = M >= 64 && N >= 1024 && Environment.ProcessorCount > 1;

        if (useParallel)
        {
            int numMTiles = (M + Mc - 1) / Mc;
            Parallel.For(0, numMTiles, mcTile =>
            {
                int mc = mcTile * Mc;
                int mcEnd = Math.Min(mc + Mc, M);

                ProcessMcTile(A, input, C, mc, mcEnd, M, N, K,
                    height, width, outHeight, outWidth,
                    kernelH, kernelW, kernelSize, strideH, strideW,
                    padH, padW, dilationH, dilationW, inChannels);
            });
        }
        else
        {
            // Sequential processing
            for (int mc = 0; mc < M; mc += Mc)
            {
                int mcEnd = Math.Min(mc + Mc, M);
                ProcessMcTile(A, input, C, mc, mcEnd, M, N, K,
                    height, width, outHeight, outWidth,
                    kernelH, kernelW, kernelSize, strideH, strideW,
                    padH, padW, dilationH, dilationW, inChannels);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void ProcessMcTile(
        float* A, float* input, float* C,
        int mc, int mcEnd, int M, int N, int K,
        int height, int width, int outHeight, int outWidth,
        int kernelH, int kernelW, int kernelSize,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW, int inChannels)
    {
        // Tile over spatial positions (N)
        for (int nc = 0; nc < N; nc += Nc)
        {
            int ncEnd = Math.Min(nc + Nc, N);

            // Tile over K (input channels * kernel elements)
            for (int kc = 0; kc < K; kc += Kc)
            {
                int kcEnd = Math.Min(kc + Kc, K);

                // Micro-kernel: process Mr x Nr blocks
                for (int m = mc; m < mcEnd; m += Mr)
                {
                    int mEnd = Math.Min(m + Mr, mcEnd);

                    for (int n = nc; n < ncEnd; n += Nr)
                    {
                        int nEnd = Math.Min(n + Nr, ncEnd);

                        // Compute micro-tile with fused im2col
                        ComputeMicroTileFused(
                            A, input, C,
                            m, mEnd, n, nEnd, kc, kcEnd, K, N,
                            height, width, outHeight, outWidth,
                            kernelH, kernelW, kernelSize, strideH, strideW,
                            padH, padW, dilationH, dilationW, inChannels);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Micro-kernel that computes a small output tile with fused im2col.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void ComputeMicroTileFused(
        float* A, float* input, float* C,
        int mStart, int mEnd, int nStart, int nEnd, int kStart, int kEnd, int K, int N,
        int height, int width, int outHeight, int outWidth,
        int kernelH, int kernelW, int kernelSize,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW, int inChannels)
    {
        // Use AVX2 for vectorized accumulation when possible
        if (UseFma && (nEnd - nStart) >= 8)
        {
            ComputeMicroTileFusedFma(
                A, input, C,
                mStart, mEnd, nStart, nEnd, kStart, kEnd, K, N,
                height, width, outHeight, outWidth,
                kernelH, kernelW, kernelSize, strideH, strideW,
                padH, padW, dilationH, dilationW, inChannels);
        }
        else
        {
            ComputeMicroTileFusedScalar(
                A, input, C,
                mStart, mEnd, nStart, nEnd, kStart, kEnd, K, N,
                height, width, outHeight, outWidth,
                kernelH, kernelW, kernelSize, strideH, strideW,
                padH, padW, dilationH, dilationW, inChannels);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void ComputeMicroTileFusedFma(
        float* A, float* input, float* C,
        int mStart, int mEnd, int nStart, int nEnd, int kStart, int kEnd, int K, int N,
        int height, int width, int outHeight, int outWidth,
        int kernelH, int kernelW, int kernelSize,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW, int inChannels)
    {
        // Pre-compute spatial positions for this N tile
        int nCount = nEnd - nStart;
        int vectorCount = nCount / 8;
        int remainder = nCount % 8;

        // Pre-allocate gather buffer outside loops to avoid CA2014
        float* gatherBuffer = stackalloc float[8];

        for (int m = mStart; m < mEnd; m++)
        {
            float* aRow = A + m * K;
            float* cRow = C + m * N;

            // Process K elements with fused im2col
            for (int k = kStart; k < kEnd; k++)
            {
                float aVal = aRow[k];
                if (Math.Abs(aVal) < 1e-10f) continue; // Skip near-zero weights

                Vector256<float> aVec = Vector256.Create(aVal);

                // Decode im2col index: k = ic * kernelSize + kh * kernelW + kw
                int ic = k / kernelSize;
                int kernelIdx = k % kernelSize;
                int kh = kernelIdx / kernelW;
                int kw = kernelIdx % kernelW;

                float* inputChannel = input + ic * height * width;

                // Vectorized processing
                for (int nv = 0; nv < vectorCount; nv++)
                {
                    int nBase = nStart + nv * 8;

                    // Gather 8 input values using fused im2col
                    Vector256<float> bVec = GatherIm2ColValues(
                        inputChannel, nBase, outWidth,
                        kh, kw, strideH, strideW, padH, padW,
                        dilationH, dilationW, height, width, gatherBuffer);

                    // Load current output, FMA, store
                    Vector256<float> cVec = Avx.LoadVector256(cRow + nBase);
                    cVec = Fma.MultiplyAdd(aVec, bVec, cVec);
                    Avx.Store(cRow + nBase, cVec);
                }

                // Handle remainder
                for (int n = nStart + vectorCount * 8; n < nEnd; n++)
                {
                    float bVal = GetIm2ColValue(inputChannel, n, outWidth,
                        kh, kw, strideH, strideW, padH, padW,
                        dilationH, dilationW, height, width);
                    cRow[n] += aVal * bVal;
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void ComputeMicroTileFusedScalar(
        float* A, float* input, float* C,
        int mStart, int mEnd, int nStart, int nEnd, int kStart, int kEnd, int K, int N,
        int height, int width, int outHeight, int outWidth,
        int kernelH, int kernelW, int kernelSize,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW, int inChannels)
    {
        for (int m = mStart; m < mEnd; m++)
        {
            float* aRow = A + m * K;
            float* cRow = C + m * N;

            for (int k = kStart; k < kEnd; k++)
            {
                float aVal = aRow[k];
                if (Math.Abs(aVal) < 1e-10f) continue;

                // Decode im2col index
                int ic = k / kernelSize;
                int kernelIdx = k % kernelSize;
                int kh = kernelIdx / kernelW;
                int kw = kernelIdx % kernelW;

                float* inputChannel = input + ic * height * width;

                for (int n = nStart; n < nEnd; n++)
                {
                    float bVal = GetIm2ColValue(inputChannel, n, outWidth,
                        kh, kw, strideH, strideW, padH, padW,
                        dilationH, dilationW, height, width);
                    cRow[n] += aVal * bVal;
                }
            }
        }
    }

    /// <summary>
    /// Gather 8 im2col values for AVX2 vectorization.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<float> GatherIm2ColValues(
        float* inputChannel, int nBase, int outWidth,
        int kh, int kw, int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW, int height, int width, float* buffer)
    {
        for (int i = 0; i < 8; i++)
        {
            int n = nBase + i;
            buffer[i] = GetIm2ColValue(inputChannel, n, outWidth,
                kh, kw, strideH, strideW, padH, padW,
                dilationH, dilationW, height, width);
        }

        return Avx.LoadVector256(buffer);
    }

    /// <summary>
    /// Computes a single im2col value on-the-fly.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float GetIm2ColValue(
        float* inputChannel, int n, int outWidth,
        int kh, int kw, int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW, int height, int width)
    {
        // Convert linear spatial index to 2D output position
        int oh = n / outWidth;
        int ow = n % outWidth;

        // Compute input position
        int ih = oh * strideH - padH + kh * dilationH;
        int iw = ow * strideW - padW + kw * dilationW;

        // Check bounds and return 0 for padding
        if (ih < 0 || ih >= height || iw < 0 || iw >= width)
        {
            return 0f;
        }

        return inputChannel[ih * width + iw];
    }
}
