using System;
using System.Buffers;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// CPU implementation of im2col (image to column) transformation for efficient convolution via GEMM.
/// Transforms input patches into columns for matrix multiplication.
/// </summary>
internal static class Im2ColHelper
{
    // Higher threshold since OpenBLAS handles parallelism for GEMM
    // Avoid thread contention between parallel im2col and multi-threaded BLAS
    private const int ParallelThreshold = 262144; // 512x512 output

    /// <summary>
    /// Performs im2col transformation on a 4D input tensor [batch, channels, height, width].
    /// Output shape: [batch, channels * kernelH * kernelW, outputH * outputW]
    /// This transforms input patches into columns for efficient GEMM-based convolution.
    /// </summary>
    public static void Im2Col(
        ReadOnlySpan<float> input,
        Span<float> output,
        int batch,
        int channels,
        int height,
        int width,
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int dilationH,
        int dilationW)
    {
        int effectiveKernelH = dilationH * (kernelH - 1) + 1;
        int effectiveKernelW = dilationW * (kernelW - 1) + 1;
        int outputH = (height + 2 * padH - effectiveKernelH) / strideH + 1;
        int outputW = (width + 2 * padW - effectiveKernelW) / strideW + 1;

        int colH = channels * kernelH * kernelW;
        int colW = outputH * outputW;
        int inputImageSize = channels * height * width;

        // Process each batch element
        for (int b = 0; b < batch; b++)
        {
            int inputOffset = b * inputImageSize;
            int outputOffset = b * colH * colW;

            Im2ColSingleImage(
                input.Slice(inputOffset, inputImageSize),
                output.Slice(outputOffset, colH * colW),
                channels, height, width,
                kernelH, kernelW,
                strideH, strideW,
                padH, padW,
                dilationH, dilationW,
                outputH, outputW);
        }
    }

    /// <summary>
    /// Performs im2col on a single image (no batch dimension).
    /// Optimized row-by-row processing for better cache utilization and SIMD.
    /// </summary>
    private static unsafe void Im2ColSingleImage(
        ReadOnlySpan<float> input,
        Span<float> output,
        int channels,
        int height,
        int width,
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int dilationH,
        int dilationW,
        int outputH,
        int outputW)
    {
        int colH = channels * kernelH * kernelW;
        int colW = outputH * outputW;

        // Fast path: stride=1, dilation=1 (most common case in CNNs)
        if (strideH == 1 && strideW == 1 && dilationH == 1 && dilationW == 1)
        {
            // Clear entire output once upfront (handles all padding)
            output.Slice(0, colH * colW).Clear();

            fixed (float* inputPtr = input)
            fixed (float* outputPtr = output)
            {
                // Calculate valid ranges that apply to all kernel positions
                int owValidStart = Math.Max(0, padW);
                int owValidEnd = Math.Min(outputW, width + padW);
                int ohValidStart = Math.Max(0, padH);
                int ohValidEnd = Math.Min(outputH, height + padH);

                int rowIdx = 0;
                for (int c = 0; c < channels; c++)
                {
                    int channelOffset = c * height * width;

                    for (int kh = 0; kh < kernelH; kh++)
                    {
                        // Adjust valid output row range for this kernel height position
                        int ohStart = Math.Max(ohValidStart, padH - kh);
                        int ohEnd = Math.Min(ohValidEnd, height + padH - kh);

                        for (int kw = 0; kw < kernelW; kw++)
                        {
                            // Adjust valid output column range for this kernel width position
                            int owStart = Math.Max(owValidStart, padW - kw);
                            int owEnd = Math.Min(owValidEnd, width + padW - kw);
                            int validWidth = owEnd - owStart;

                            if (validWidth > 0 && ohEnd > ohStart)
                            {
                                float* outRow = outputPtr + rowIdx * colW;

                                for (int oh = ohStart; oh < ohEnd; oh++)
                                {
                                    int ih = oh + kh - padH;
                                    int inputStart = channelOffset + ih * width + (owStart + kw - padW);
                                    int outputStart = oh * outputW + owStart;

                                    Buffer.MemoryCopy(
                                        inputPtr + inputStart,
                                        outRow + outputStart,
                                        validWidth * sizeof(float),
                                        validWidth * sizeof(float));
                                }
                            }

                            rowIdx++;
                        }
                    }
                }
            }
        }
        else
        {
            // General path for arbitrary stride/dilation
            fixed (float* inputPtr = input)
            fixed (float* outputPtr = output)
            {
                int rowIdx = 0;
                for (int c = 0; c < channels; c++)
                {
                    int channelOffset = c * height * width;

                    for (int kh = 0; kh < kernelH; kh++)
                    {
                        for (int kw = 0; kw < kernelW; kw++)
                        {
                            float* outRow = outputPtr + rowIdx * colW;
                            Im2ColRowGeneral(inputPtr, outRow, channelOffset,
                                height, width, kh, kw, strideH, strideW,
                                padH, padW, dilationH, dilationW, outputH, outputW);
                            rowIdx++;
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Optimized row processing for stride=1, dilation=1 case.
    /// Uses bulk memory copies for contiguous regions instead of element-by-element access.
    /// </summary>
    private static unsafe void Im2ColRowOptimized(
        float* input,
        float* outRow,
        int channelOffset,
        int height,
        int width,
        int kh,
        int kw,
        int padH,
        int padW,
        int outputH,
        int outputW)
    {
        // For stride=1, dilation=1: output position (oh, ow) maps to input position (ih, iw) = (oh + kh - padH, ow + kw - padW)
        // We can use bulk copies for contiguous valid regions

        // Calculate valid input row range
        int ihStart = kh - padH;  // ih when oh = 0
        int ohValidStart = Math.Max(0, padH - kh);  // First oh where ih >= 0
        int ohValidEnd = Math.Min(outputH, height + padH - kh);  // First oh where ih >= height

        // Calculate valid input column range for each output row
        int iwStart = kw - padW;  // iw when ow = 0
        int owValidStart = Math.Max(0, padW - kw);  // First ow where iw >= 0
        int owValidEnd = Math.Min(outputW, width + padW - kw);  // First ow where iw >= width
        int validWidth = owValidEnd - owValidStart;

        // Zero the entire output first (handles padding efficiently)
        new Span<float>(outRow, outputH * outputW).Clear();

        // Process valid rows with bulk copy
        if (validWidth > 0 && ohValidEnd > ohValidStart)
        {
            for (int oh = ohValidStart; oh < ohValidEnd; oh++)
            {
                int ih = oh + kh - padH;
                int inputRowOffset = channelOffset + ih * width;
                int outputRowOffset = oh * outputW;

                // Bulk copy the valid portion of this row
                int inputStart = inputRowOffset + (owValidStart + kw - padW);
                int outputStart = outputRowOffset + owValidStart;

                Buffer.MemoryCopy(
                    input + inputStart,
                    outRow + outputStart,
                    validWidth * sizeof(float),
                    validWidth * sizeof(float));
            }
        }
    }

    /// <summary>
    /// General row processing for arbitrary stride and dilation.
    /// </summary>
    private static unsafe void Im2ColRowGeneral(
        float* input,
        float* outRow,
        int channelOffset,
        int height,
        int width,
        int kh,
        int kw,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int dilationH,
        int dilationW,
        int outputH,
        int outputW)
    {
        int colIdx = 0;

        for (int oh = 0; oh < outputH; oh++)
        {
            int ih = oh * strideH + kh * dilationH - padH;

            for (int ow = 0; ow < outputW; ow++)
            {
                int iw = ow * strideW + kw * dilationW - padW;

                float val = 0f;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                {
                    val = input[channelOffset + ih * width + iw];
                }

                outRow[colIdx++] = val;
            }
        }
    }

    private static void ProcessColumnArray(
        float[] input,
        float[] output,
        int channels,
        int height,
        int width,
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int dilationH,
        int dilationW,
        int colH,
        int colW,
        int oh,
        int ow,
        int colIdx)
    {
        int rowIdx = 0;

        for (int c = 0; c < channels; c++)
        {
            for (int kh = 0; kh < kernelH; kh++)
            {
                for (int kw = 0; kw < kernelW; kw++)
                {
                    int ih = oh * strideH + kh * dilationH - padH;
                    int iw = ow * strideW + kw * dilationW - padW;

                    float val = 0f;
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                    {
                        int inputIdx = c * height * width + ih * width + iw;
                        val = input[inputIdx];
                    }

                    // Column-major layout for efficient GEMM
                    output[rowIdx * colW + colIdx] = val;
                    rowIdx++;
                }
            }
        }
    }

    private static void ProcessColumnSpan(
        ReadOnlySpan<float> input,
        Span<float> output,
        int channels,
        int height,
        int width,
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int dilationH,
        int dilationW,
        int colH,
        int colW,
        int oh,
        int ow,
        int colIdx)
    {
        int rowIdx = 0;

        for (int c = 0; c < channels; c++)
        {
            for (int kh = 0; kh < kernelH; kh++)
            {
                for (int kw = 0; kw < kernelW; kw++)
                {
                    int ih = oh * strideH + kh * dilationH - padH;
                    int iw = ow * strideW + kw * dilationW - padW;

                    float val = 0f;
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                    {
                        int inputIdx = c * height * width + ih * width + iw;
                        val = input[inputIdx];
                    }

                    // Column-major layout for efficient GEMM
                    output[rowIdx * colW + colIdx] = val;
                    rowIdx++;
                }
            }
        }
    }

    /// <summary>
    /// Performs Conv2D using im2col + GEMM approach.
    /// This is significantly faster than naive nested loops for large convolutions.
    /// </summary>
    /// <returns>True if GEMM was used successfully, false if fallback is needed</returns>
    public static bool TryConv2DWithGemm(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> kernel,
        Span<float> output,
        int batch,
        int inChannels,
        int height,
        int width,
        int outChannels,
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int dilationH,
        int dilationW)
    {
        int effectiveKernelH = dilationH * (kernelH - 1) + 1;
        int effectiveKernelW = dilationW * (kernelW - 1) + 1;
        int outputH = (height + 2 * padH - effectiveKernelH) / strideH + 1;
        int outputW = (width + 2 * padW - effectiveKernelW) / strideW + 1;

        // im2col matrix dimensions
        int colH = inChannels * kernelH * kernelW;  // M for kernel, K for GEMM
        int colW = outputH * outputW;               // N for GEMM

        // GEMM: kernel (outChannels x colH) @ im2col (colH x colW) = output (outChannels x colW)
        // M = outChannels, K = colH, N = colW

        // Allocate im2col buffer
        var pool = ArrayPool<float>.Shared;
        float[] im2colBuffer = pool.Rent(batch * colH * colW);

        try
        {
            // Step 1: Convert input to im2col format
            Im2Col(input, im2colBuffer.AsSpan(0, batch * colH * colW),
                batch, inChannels, height, width,
                kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);

            // Step 2: GEMM for each batch element
            // Kernel is reshaped: [outChannels, inChannels, kernelH, kernelW] -> [outChannels, colH]
            // Already in correct layout for GEMM

            for (int b = 0; b < batch; b++)
            {
                int im2colOffset = b * colH * colW;
                int outputOffset = b * outChannels * outputH * outputW;

                // Use span-based GEMM to avoid array copies
                bool usedBlas = BlasProvider.TryGemm(
                    outChannels, colW, colH,
                    kernel,  // ReadOnlySpan<float>
                    colH,
                    im2colBuffer.AsSpan(im2colOffset, colH * colW),  // ReadOnlySpan<float>
                    colW,
                    output.Slice(outputOffset, outChannels * colW),  // Span<float>
                    colW);

                if (!usedBlas)
                {
                    // Fallback to blocked matrix multiply
                    MultiplyMatrixBlocked(
                        kernel,
                        im2colBuffer.AsSpan(im2colOffset, colH * colW),
                        output.Slice(outputOffset, outChannels * colW),
                        outChannels, colH, colW);
                }
            }

            return true;
        }
        finally
        {
            pool.Return(im2colBuffer);
        }
    }

    /// <summary>
    /// Blocked matrix multiplication fallback when BLAS is not available.
    /// C = A @ B where A is [m, k], B is [k, n], C is [m, n]
    /// </summary>
    private static void MultiplyMatrixBlocked(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m,
        int k,
        int n)
    {
        const int BlockSize = 64;

        // Initialize output to zero
        c.Clear();

        // Blocked matrix multiplication with cache-friendly access pattern
        for (int ii = 0; ii < m; ii += BlockSize)
        {
            int iEnd = Math.Min(ii + BlockSize, m);

            for (int kk = 0; kk < k; kk += BlockSize)
            {
                int kEnd = Math.Min(kk + BlockSize, k);

                for (int jj = 0; jj < n; jj += BlockSize)
                {
                    int jEnd = Math.Min(jj + BlockSize, n);

                    // Process block
                    for (int i = ii; i < iEnd; i++)
                    {
                        for (int kIdx = kk; kIdx < kEnd; kIdx++)
                        {
                            float aik = a[i * k + kIdx];
                            int bRowOffset = kIdx * n + jj;
                            int cRowOffset = i * n + jj;

                            for (int j = 0; j < jEnd - jj; j++)
                            {
                                c[cRowOffset + j] += aik * b[bRowOffset + j];
                            }
                        }
                    }
                }
            }
        }
    }
}
