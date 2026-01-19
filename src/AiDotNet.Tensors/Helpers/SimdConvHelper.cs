using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// SIMD-optimized convolution kernels using AVX2/AVX-512 intrinsics.
/// Provides highly optimized direct convolution for common kernel sizes.
/// </summary>
internal static class SimdConvHelper
{
    private static readonly bool UseAvx512 = Avx512F.IsSupported;
    private static readonly bool UseAvx2 = Avx2.IsSupported;
    private static readonly bool UseFma = Fma.IsSupported;
    private static readonly bool UseSse41 = Sse41.IsSupported;

    // Minimum output size for parallel processing (lowered for more parallelism)
    private const int ParallelThreshold = 1024; // 32x32 output

    // Output channel block size for better cache utilization
    private const int ChannelBlockSize = 4;

    /// <summary>
    /// Check if SIMD-optimized convolution is available for this configuration.
    /// </summary>
    public static bool CanUseSimdConv(int kernelH, int kernelW, int strideH, int strideW)
    {
        // SIMD convolution requires AVX2 and works best with stride=1
        if (!UseAvx2)
        {
            return false;
        }

        // Optimized for common kernel sizes with stride=1
        return (kernelH == 3 && kernelW == 3 && strideH == 1 && strideW == 1) ||
               (kernelH == 1 && kernelW == 1 && strideH == 1 && strideW == 1);
    }

    /// <summary>
    /// Performs 3x3 convolution with stride=1 using AVX2 intrinsics.
    /// </summary>
    public static unsafe void Conv3x3Stride1(
        float* input, float* kernel, float* output,
        int batch, int inChannels, int height, int width,
        int outChannels, int padH, int padW, int dilationH, int dilationW)
    {
        int outHeight = height + 2 * padH - (dilationH * 2 + 1) + 1;
        int outWidth = width + 2 * padW - (dilationW * 2 + 1) + 1;
        int outputSize = outHeight * outWidth;

        bool useParallel = outputSize >= ParallelThreshold && Environment.ProcessorCount > 1;

        for (int b = 0; b < batch; b++)
        {
            float* inputBatch = input + b * inChannels * height * width;
            float* outputBatch = output + b * outChannels * outHeight * outWidth;

            if (useParallel)
            {
                Parallel.For(0, outChannels, oc =>
                {
                    Conv3x3Stride1SingleChannel(
                        inputBatch, kernel + oc * inChannels * 9,
                        outputBatch + oc * outputSize,
                        inChannels, height, width, outHeight, outWidth,
                        padH, padW, dilationH, dilationW);
                });
            }
            else
            {
                for (int oc = 0; oc < outChannels; oc++)
                {
                    Conv3x3Stride1SingleChannel(
                        inputBatch, kernel + oc * inChannels * 9,
                        outputBatch + oc * outputSize,
                        inChannels, height, width, outHeight, outWidth,
                        padH, padW, dilationH, dilationW);
                }
            }
        }
    }

    /// <summary>
    /// Performs 1x1 convolution (pointwise) using AVX2 intrinsics.
    /// </summary>
    public static unsafe void Conv1x1(
        float* input, float* kernel, float* output,
        int batch, int inChannels, int height, int width, int outChannels)
    {
        int spatialSize = height * width;

        for (int b = 0; b < batch; b++)
        {
            float* inputBatch = input + b * inChannels * spatialSize;
            float* outputBatch = output + b * outChannels * spatialSize;

            // 1x1 conv is essentially matrix multiply: [outChannels, inChannels] @ [inChannels, spatialSize]
            // Use GEMM-like approach with AVX2
            Conv1x1Gemm(inputBatch, kernel, outputBatch, outChannels, inChannels, spatialSize);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Conv3x3Stride1SingleChannel(
        float* input, float* kernelOc, float* outputChannel,
        int inChannels, int height, int width, int outHeight, int outWidth,
        int padH, int padW, int dilationH, int dilationW)
    {
        // Clear output
        int outputSize = outHeight * outWidth;
        new Span<float>(outputChannel, outputSize).Clear();

        // Process each input channel
        for (int ic = 0; ic < inChannels; ic++)
        {
            float* inputChannel = input + ic * height * width;
            float* kernelChannel = kernelOc + ic * 9;

            // Load 3x3 kernel into registers
            if (UseFma)
            {
                Conv3x3SingleChannelFma(inputChannel, kernelChannel, outputChannel,
                    height, width, outHeight, outWidth, padH, padW, dilationH, dilationW);
            }
            else
            {
                Conv3x3SingleChannelAvx2(inputChannel, kernelChannel, outputChannel,
                    height, width, outHeight, outWidth, padH, padW, dilationH, dilationW);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Conv3x3SingleChannelFma(
        float* input, float* kernel, float* output,
        int height, int width, int outHeight, int outWidth,
        int padH, int padW, int dilationH, int dilationW)
    {
        // Load kernel values into SIMD registers (broadcast)
        Vector256<float> k00 = Vector256.Create(kernel[0]);
        Vector256<float> k01 = Vector256.Create(kernel[1]);
        Vector256<float> k02 = Vector256.Create(kernel[2]);
        Vector256<float> k10 = Vector256.Create(kernel[3]);
        Vector256<float> k11 = Vector256.Create(kernel[4]);
        Vector256<float> k12 = Vector256.Create(kernel[5]);
        Vector256<float> k20 = Vector256.Create(kernel[6]);
        Vector256<float> k21 = Vector256.Create(kernel[7]);
        Vector256<float> k22 = Vector256.Create(kernel[8]);

        // Pre-allocate boundary handling buffer outside loop to avoid CA2014
        float* boundaryBuffer = stackalloc float[8];

        // Fast path for no-dilation case (most common)
        if (dilationH == 1 && dilationW == 1)
        {
            // Process interior rows (no top/bottom boundary)
            int ohStart = padH > 0 ? padH : 0;
            int ohEnd = outHeight - (padH > 0 ? padH : 0);
            if (ohEnd > outHeight) ohEnd = outHeight;

            // Handle boundary rows at top
            for (int topOh = 0; topOh < ohStart && topOh < outHeight; topOh++)
            {
                ProcessBoundaryRow(input, kernel, output + topOh * outWidth,
                    height, width, outWidth, topOh, padH, padW, boundaryBuffer);
            }

            // Fast interior processing with fully unrolled kernel
            // Process 2 output rows at a time for better instruction-level parallelism
            int oh = ohStart;
            for (; oh + 1 < ohEnd; oh += 2)
            {
                int ih0 = oh - padH;
                float* inputRow0 = input + ih0 * width;
                float* inputRow1 = input + (ih0 + 1) * width;
                float* inputRow2 = input + (ih0 + 2) * width;
                float* inputRow3 = input + (ih0 + 3) * width;
                float* outputRow0 = output + oh * outWidth;
                float* outputRow1 = output + (oh + 1) * outWidth;

                // Interior columns (no boundary handling needed)
                int owStart = padW > 0 ? padW : 0;
                int owEnd = outWidth - (padW > 0 ? padW : 0) - 7;

                // Handle left boundary columns
                for (int leftOw = 0; leftOw < owStart && leftOw < outWidth; leftOw++)
                {
                    outputRow0[leftOw] += ComputeScalarConv3x3(input, kernel, height, width, oh, leftOw, padH, padW);
                    outputRow1[leftOw] += ComputeScalarConv3x3(input, kernel, height, width, oh + 1, leftOw, padH, padW);
                }

                // Process 8 elements at a time in the interior - 2 rows simultaneously
                int ow = owStart;
                for (; ow < owEnd; ow += 8)
                {
                    int iw = ow - padW;

                    // Prefetch next cache line (64 bytes ahead = 16 floats)
                    if (UseSse41 && ow + 16 < owEnd)
                    {
                        Sse.Prefetch1(inputRow0 + iw + 16);
                        Sse.Prefetch1(inputRow1 + iw + 16);
                        Sse.Prefetch1(inputRow2 + iw + 16);
                        Sse.Prefetch1(inputRow3 + iw + 16);
                    }

                    // Load vectors for row 0 output
                    Vector256<float> r0_0 = Avx.LoadVector256(inputRow0 + iw);
                    Vector256<float> r0_1 = Avx.LoadVector256(inputRow0 + iw + 1);
                    Vector256<float> r0_2 = Avx.LoadVector256(inputRow0 + iw + 2);
                    Vector256<float> r1_0 = Avx.LoadVector256(inputRow1 + iw);
                    Vector256<float> r1_1 = Avx.LoadVector256(inputRow1 + iw + 1);
                    Vector256<float> r1_2 = Avx.LoadVector256(inputRow1 + iw + 2);
                    Vector256<float> r2_0 = Avx.LoadVector256(inputRow2 + iw);
                    Vector256<float> r2_1 = Avx.LoadVector256(inputRow2 + iw + 1);
                    Vector256<float> r2_2 = Avx.LoadVector256(inputRow2 + iw + 2);

                    // Convolution for first output row
                    Vector256<float> acc0 = Fma.MultiplyAdd(r0_0, k00, Vector256<float>.Zero);
                    acc0 = Fma.MultiplyAdd(r0_1, k01, acc0);
                    acc0 = Fma.MultiplyAdd(r0_2, k02, acc0);
                    acc0 = Fma.MultiplyAdd(r1_0, k10, acc0);
                    acc0 = Fma.MultiplyAdd(r1_1, k11, acc0);
                    acc0 = Fma.MultiplyAdd(r1_2, k12, acc0);
                    acc0 = Fma.MultiplyAdd(r2_0, k20, acc0);
                    acc0 = Fma.MultiplyAdd(r2_1, k21, acc0);
                    acc0 = Fma.MultiplyAdd(r2_2, k22, acc0);

                    // Load additional row for second output
                    Vector256<float> r3_0 = Avx.LoadVector256(inputRow3 + iw);
                    Vector256<float> r3_1 = Avx.LoadVector256(inputRow3 + iw + 1);
                    Vector256<float> r3_2 = Avx.LoadVector256(inputRow3 + iw + 2);

                    // Convolution for second output row (reuse r1, r2 rows)
                    Vector256<float> acc1 = Fma.MultiplyAdd(r1_0, k00, Vector256<float>.Zero);
                    acc1 = Fma.MultiplyAdd(r1_1, k01, acc1);
                    acc1 = Fma.MultiplyAdd(r1_2, k02, acc1);
                    acc1 = Fma.MultiplyAdd(r2_0, k10, acc1);
                    acc1 = Fma.MultiplyAdd(r2_1, k11, acc1);
                    acc1 = Fma.MultiplyAdd(r2_2, k12, acc1);
                    acc1 = Fma.MultiplyAdd(r3_0, k20, acc1);
                    acc1 = Fma.MultiplyAdd(r3_1, k21, acc1);
                    acc1 = Fma.MultiplyAdd(r3_2, k22, acc1);

                    // Store both output rows
                    Vector256<float> current0 = Avx.LoadVector256(outputRow0 + ow);
                    Vector256<float> current1 = Avx.LoadVector256(outputRow1 + ow);
                    Avx.Store(outputRow0 + ow, Avx.Add(current0, acc0));
                    Avx.Store(outputRow1 + ow, Avx.Add(current1, acc1));
                }

                // Handle right boundary columns
                for (; ow < outWidth; ow++)
                {
                    outputRow0[ow] += ComputeScalarConv3x3(input, kernel, height, width, oh, ow, padH, padW);
                    outputRow1[ow] += ComputeScalarConv3x3(input, kernel, height, width, oh + 1, ow, padH, padW);
                }
            }

            // Handle remaining single row if ohEnd - ohStart is odd
            for (; oh < ohEnd; oh++)
            {
                int ih0 = oh - padH;
                float* inputRow0 = input + ih0 * width;
                float* inputRow1 = input + (ih0 + 1) * width;
                float* inputRow2 = input + (ih0 + 2) * width;
                float* outputRow = output + oh * outWidth;

                int owStart = padW > 0 ? padW : 0;
                int owEnd = outWidth - (padW > 0 ? padW : 0) - 7;

                for (int leftOw = 0; leftOw < owStart && leftOw < outWidth; leftOw++)
                {
                    outputRow[leftOw] += ComputeScalarConv3x3(input, kernel, height, width, oh, leftOw, padH, padW);
                }

                int ow2 = owStart;
                for (; ow2 < owEnd; ow2 += 8)
                {
                    int iw = ow2 - padW;
                    Vector256<float> r0_0 = Avx.LoadVector256(inputRow0 + iw);
                    Vector256<float> r0_1 = Avx.LoadVector256(inputRow0 + iw + 1);
                    Vector256<float> r0_2 = Avx.LoadVector256(inputRow0 + iw + 2);
                    Vector256<float> r1_0 = Avx.LoadVector256(inputRow1 + iw);
                    Vector256<float> r1_1 = Avx.LoadVector256(inputRow1 + iw + 1);
                    Vector256<float> r1_2 = Avx.LoadVector256(inputRow1 + iw + 2);
                    Vector256<float> r2_0 = Avx.LoadVector256(inputRow2 + iw);
                    Vector256<float> r2_1 = Avx.LoadVector256(inputRow2 + iw + 1);
                    Vector256<float> r2_2 = Avx.LoadVector256(inputRow2 + iw + 2);

                    Vector256<float> acc = Fma.MultiplyAdd(r0_0, k00, Vector256<float>.Zero);
                    acc = Fma.MultiplyAdd(r0_1, k01, acc);
                    acc = Fma.MultiplyAdd(r0_2, k02, acc);
                    acc = Fma.MultiplyAdd(r1_0, k10, acc);
                    acc = Fma.MultiplyAdd(r1_1, k11, acc);
                    acc = Fma.MultiplyAdd(r1_2, k12, acc);
                    acc = Fma.MultiplyAdd(r2_0, k20, acc);
                    acc = Fma.MultiplyAdd(r2_1, k21, acc);
                    acc = Fma.MultiplyAdd(r2_2, k22, acc);

                    Vector256<float> current = Avx.LoadVector256(outputRow + ow2);
                    Avx.Store(outputRow + ow2, Avx.Add(current, acc));
                }

                for (; ow2 < outWidth; ow2++)
                {
                    outputRow[ow2] += ComputeScalarConv3x3(input, kernel, height, width, oh, ow2, padH, padW);
                }
            }

            // Handle boundary rows at bottom
            for (int bottomOh = ohEnd; bottomOh < outHeight; bottomOh++)
            {
                ProcessBoundaryRow(input, kernel, output + bottomOh * outWidth,
                    height, width, outWidth, bottomOh, padH, padW, boundaryBuffer);
            }
        }
        else
        {
            // Dilation case - use original general loop
            Conv3x3SingleChannelFmaWithDilation(input, kernel, output,
                height, width, outHeight, outWidth, padH, padW, dilationH, dilationW, boundaryBuffer);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float ComputeScalarConv3x3(float* input, float* kernel,
        int height, int width, int oh, int ow, int padH, int padW)
    {
        float sum = 0f;
        int ihBase = oh - padH;
        int iwBase = ow - padW;

        for (int kh = 0; kh < 3; kh++)
        {
            int ih = ihBase + kh;
            if (ih < 0 || ih >= height) continue;

            for (int kw = 0; kw < 3; kw++)
            {
                int iw = iwBase + kw;
                if (iw >= 0 && iw < width)
                {
                    sum += input[ih * width + iw] * kernel[kh * 3 + kw];
                }
            }
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void ProcessBoundaryRow(float* input, float* kernel, float* outputRow,
        int height, int width, int outWidth, int oh, int padH, int padW, float* boundaryBuffer)
    {
        for (int ow = 0; ow < outWidth; ow++)
        {
            outputRow[ow] += ComputeScalarConv3x3(input, kernel, height, width, oh, ow, padH, padW);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Conv3x3SingleChannelFmaWithDilation(
        float* input, float* kernel, float* output,
        int height, int width, int outHeight, int outWidth,
        int padH, int padW, int dilationH, int dilationW, float* boundaryBuffer)
    {
        // General case with dilation support
        for (int oh = 0; oh < outHeight; oh++)
        {
            int ihBase = oh - padH;
            float* outputRow = output + oh * outWidth;

            int ow = 0;
            int vectorEnd = outWidth - 7;

            for (; ow < vectorEnd; ow += 8)
            {
                Vector256<float> acc = Vector256.Create(0f);
                int iwBase = ow - padW;

                for (int kh = 0; kh < 3; kh++)
                {
                    int ih = ihBase + kh * dilationH;
                    if (ih < 0 || ih >= height) continue;

                    float* inputRow = input + ih * width;

                    for (int kw = 0; kw < 3; kw++)
                    {
                        int iwStart = iwBase + kw * dilationW;
                        float kVal = kernel[kh * 3 + kw];

                        Vector256<float> inputVec;
                        if (iwStart >= 0 && iwStart + 7 < width)
                        {
                            inputVec = Avx.LoadVector256(inputRow + iwStart);
                        }
                        else
                        {
                            for (int i = 0; i < 8; i++)
                            {
                                int iw = iwStart + i;
                                boundaryBuffer[i] = (iw >= 0 && iw < width) ? inputRow[iw] : 0f;
                            }
                            inputVec = Avx.LoadVector256(boundaryBuffer);
                        }

                        Vector256<float> kVec = Vector256.Create(kVal);
                        acc = Fma.MultiplyAdd(inputVec, kVec, acc);
                    }
                }

                Vector256<float> current = Avx.LoadVector256(outputRow + ow);
                Avx.Store(outputRow + ow, Avx.Add(current, acc));
            }

            for (; ow < outWidth; ow++)
            {
                outputRow[ow] += ComputeScalarConv3x3WithDilation(input, kernel, height, width, oh, ow, padH, padW, dilationH, dilationW);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float ComputeScalarConv3x3WithDilation(float* input, float* kernel,
        int height, int width, int oh, int ow, int padH, int padW, int dilationH, int dilationW)
    {
        float sum = 0f;
        int ihBase = oh - padH;
        int iwBase = ow - padW;

        for (int kh = 0; kh < 3; kh++)
        {
            int ih = ihBase + kh * dilationH;
            if (ih < 0 || ih >= height) continue;

            for (int kw = 0; kw < 3; kw++)
            {
                int iw = iwBase + kw * dilationW;
                if (iw >= 0 && iw < width)
                {
                    sum += input[ih * width + iw] * kernel[kh * 3 + kw];
                }
            }
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Conv3x3SingleChannelAvx2(
        float* input, float* kernel, float* output,
        int height, int width, int outHeight, int outWidth,
        int padH, int padW, int dilationH, int dilationW)
    {
        // Load kernel values into SIMD registers (broadcast)
        Vector256<float> k00 = Vector256.Create(kernel[0]);
        Vector256<float> k01 = Vector256.Create(kernel[1]);
        Vector256<float> k02 = Vector256.Create(kernel[2]);
        Vector256<float> k10 = Vector256.Create(kernel[3]);
        Vector256<float> k11 = Vector256.Create(kernel[4]);
        Vector256<float> k12 = Vector256.Create(kernel[5]);
        Vector256<float> k20 = Vector256.Create(kernel[6]);
        Vector256<float> k21 = Vector256.Create(kernel[7]);
        Vector256<float> k22 = Vector256.Create(kernel[8]);

        // Pre-allocate boundary handling buffer outside loop to avoid CA2014
        float* boundaryBuffer = stackalloc float[8];

        // Fast path for no-dilation case (most common)
        if (dilationH == 1 && dilationW == 1)
        {
            // Process interior rows (no top/bottom boundary)
            int ohStart = padH > 0 ? padH : 0;
            int ohEnd = outHeight - (padH > 0 ? padH : 0);
            if (ohEnd > outHeight) ohEnd = outHeight;

            // Handle boundary rows at top
            for (int oh = 0; oh < ohStart && oh < outHeight; oh++)
            {
                ProcessBoundaryRow(input, kernel, output + oh * outWidth,
                    height, width, outWidth, oh, padH, padW, boundaryBuffer);
            }

            // Fast interior processing with fully unrolled kernel
            for (int oh = ohStart; oh < ohEnd; oh++)
            {
                int ih0 = oh - padH;
                float* inputRow0 = input + ih0 * width;
                float* inputRow1 = input + (ih0 + 1) * width;
                float* inputRow2 = input + (ih0 + 2) * width;
                float* outputRow = output + oh * outWidth;

                // Interior columns (no boundary handling needed)
                int owStart = padW > 0 ? padW : 0;
                int owEnd = outWidth - (padW > 0 ? padW : 0) - 7;

                // Handle left boundary columns
                for (int leftOw = 0; leftOw < owStart && leftOw < outWidth; leftOw++)
                {
                    outputRow[leftOw] += ComputeScalarConv3x3(input, kernel, height, width, oh, leftOw, padH, padW);
                }

                // Process 8 elements at a time in the interior
                int ow = owStart;
                for (; ow < owEnd; ow += 8)
                {
                    int iw = ow - padW;

                    // Load vectors from each input row
                    Vector256<float> r0_0 = Avx.LoadVector256(inputRow0 + iw);
                    Vector256<float> r0_1 = Avx.LoadVector256(inputRow0 + iw + 1);
                    Vector256<float> r0_2 = Avx.LoadVector256(inputRow0 + iw + 2);
                    Vector256<float> r1_0 = Avx.LoadVector256(inputRow1 + iw);
                    Vector256<float> r1_1 = Avx.LoadVector256(inputRow1 + iw + 1);
                    Vector256<float> r1_2 = Avx.LoadVector256(inputRow1 + iw + 2);
                    Vector256<float> r2_0 = Avx.LoadVector256(inputRow2 + iw);
                    Vector256<float> r2_1 = Avx.LoadVector256(inputRow2 + iw + 1);
                    Vector256<float> r2_2 = Avx.LoadVector256(inputRow2 + iw + 2);

                    // Fully unrolled 3x3 convolution without FMA
                    Vector256<float> acc = Avx.Multiply(r0_0, k00);
                    acc = Avx.Add(acc, Avx.Multiply(r0_1, k01));
                    acc = Avx.Add(acc, Avx.Multiply(r0_2, k02));
                    acc = Avx.Add(acc, Avx.Multiply(r1_0, k10));
                    acc = Avx.Add(acc, Avx.Multiply(r1_1, k11));
                    acc = Avx.Add(acc, Avx.Multiply(r1_2, k12));
                    acc = Avx.Add(acc, Avx.Multiply(r2_0, k20));
                    acc = Avx.Add(acc, Avx.Multiply(r2_1, k21));
                    acc = Avx.Add(acc, Avx.Multiply(r2_2, k22));

                    // Add to output
                    Vector256<float> current = Avx.LoadVector256(outputRow + ow);
                    Avx.Store(outputRow + ow, Avx.Add(current, acc));
                }

                // Handle right boundary columns
                for (; ow < outWidth; ow++)
                {
                    outputRow[ow] += ComputeScalarConv3x3(input, kernel, height, width, oh, ow, padH, padW);
                }
            }

            // Handle boundary rows at bottom
            for (int oh = ohEnd; oh < outHeight; oh++)
            {
                ProcessBoundaryRow(input, kernel, output + oh * outWidth,
                    height, width, outWidth, oh, padH, padW, boundaryBuffer);
            }
        }
        else
        {
            // Dilation case - use general loop
            Conv3x3SingleChannelAvx2WithDilation(input, kernel, output,
                height, width, outHeight, outWidth, padH, padW, dilationH, dilationW, boundaryBuffer);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Conv3x3SingleChannelAvx2WithDilation(
        float* input, float* kernel, float* output,
        int height, int width, int outHeight, int outWidth,
        int padH, int padW, int dilationH, int dilationW, float* boundaryBuffer)
    {
        // General case with dilation support
        for (int oh = 0; oh < outHeight; oh++)
        {
            int ihBase = oh - padH;
            float* outputRow = output + oh * outWidth;

            int ow = 0;
            int vectorEnd = outWidth - 7;

            for (; ow < vectorEnd; ow += 8)
            {
                Vector256<float> acc = Vector256.Create(0f);
                int iwBase = ow - padW;

                for (int kh = 0; kh < 3; kh++)
                {
                    int ih = ihBase + kh * dilationH;
                    if (ih < 0 || ih >= height) continue;

                    float* inputRow = input + ih * width;

                    for (int kw = 0; kw < 3; kw++)
                    {
                        int iwStart = iwBase + kw * dilationW;
                        float kVal = kernel[kh * 3 + kw];

                        Vector256<float> inputVec;
                        if (iwStart >= 0 && iwStart + 7 < width)
                        {
                            inputVec = Avx.LoadVector256(inputRow + iwStart);
                        }
                        else
                        {
                            for (int i = 0; i < 8; i++)
                            {
                                int iw = iwStart + i;
                                boundaryBuffer[i] = (iw >= 0 && iw < width) ? inputRow[iw] : 0f;
                            }
                            inputVec = Avx.LoadVector256(boundaryBuffer);
                        }

                        Vector256<float> kVec = Vector256.Create(kVal);
                        acc = Avx.Add(acc, Avx.Multiply(inputVec, kVec));
                    }
                }

                Vector256<float> current = Avx.LoadVector256(outputRow + ow);
                Avx.Store(outputRow + ow, Avx.Add(current, acc));
            }

            for (; ow < outWidth; ow++)
            {
                outputRow[ow] += ComputeScalarConv3x3WithDilation(input, kernel, height, width, oh, ow, padH, padW, dilationH, dilationW);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Conv1x1Gemm(
        float* input, float* kernel, float* output,
        int outChannels, int inChannels, int spatialSize)
    {
        // 1x1 conv is GEMM: output[oc, spatial] = sum_ic(kernel[oc, ic] * input[ic, spatial])
        const int BlockSize = 32;

        // Clear output
        new Span<float>(output, outChannels * spatialSize).Clear();

        // Block over output channels
        for (int ocBlock = 0; ocBlock < outChannels; ocBlock += BlockSize)
        {
            int ocEnd = Math.Min(ocBlock + BlockSize, outChannels);

            // Block over input channels
            for (int icBlock = 0; icBlock < inChannels; icBlock += BlockSize)
            {
                int icEnd = Math.Min(icBlock + BlockSize, inChannels);

                // Process this block
                for (int oc = ocBlock; oc < ocEnd; oc++)
                {
                    float* outputChannel = output + oc * spatialSize;
                    float* kernelRow = kernel + oc * inChannels;

                    for (int ic = icBlock; ic < icEnd; ic++)
                    {
                        float kVal = kernelRow[ic];
                        float* inputChannel = input + ic * spatialSize;

                        Vector256<float> kVec = Vector256.Create(kVal);

                        int s = 0;
                        int vectorEnd = spatialSize - 7;

                        for (; s < vectorEnd; s += 8)
                        {
                            Vector256<float> inVec = Avx.LoadVector256(inputChannel + s);
                            Vector256<float> outVec = Avx.LoadVector256(outputChannel + s);

                            if (UseFma)
                            {
                                Avx.Store(outputChannel + s, Fma.MultiplyAdd(inVec, kVec, outVec));
                            }
                            else
                            {
                                Avx.Store(outputChannel + s, Avx.Add(outVec, Avx.Multiply(inVec, kVec)));
                            }
                        }

                        for (; s < spatialSize; s++)
                        {
                            outputChannel[s] += kVal * inputChannel[s];
                        }
                    }
                }
            }
        }
    }
}
