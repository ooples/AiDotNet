using System;
using System.Runtime.CompilerServices;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Direct convolution implementation optimized for 3x3 kernels.
/// Uses SIMD (AVX2) vectorization for high performance without im2col memory overhead.
/// </summary>
internal static class DirectConv2DHelper
{
    /// <summary>
    /// Check if direct convolution is beneficial for these parameters.
    /// </summary>
    public static bool ShouldUseDirectConv(int kernelH, int kernelW, int strideH, int strideW, int dilationH, int dilationW)
    {
        // Direct conv is most beneficial for small kernels with stride=1, dilation=1
        return kernelH <= 5 && kernelW <= 5 && strideH == 1 && strideW == 1 && dilationH == 1 && dilationW == 1;
    }

    /// <summary>
    /// Performs 3x3 direct convolution with padding, optimized with SIMD.
    /// </summary>
    public static void Conv2D3x3(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> kernel,
        Span<float> output,
        int batch,
        int inChannels,
        int height,
        int width,
        int outChannels,
        int padH,
        int padW)
    {
        int outputH = height + 2 * padH - 2;
        int outputW = width + 2 * padW - 2;
        int outputSize = outChannels * outputH * outputW;

        // Clear output first
        output.Slice(0, batch * outputSize).Clear();

        // Use sequential SIMD-optimized implementation
        // Parallel version has overhead that doesn't help for typical CNN sizes
        Conv2D3x3Sequential(input, kernel, output, batch, inChannels, height, width, outChannels, padH, padW, outputH, outputW);
    }

    private static void Conv2D3x3Sequential(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> kernel,
        Span<float> output,
        int batch,
        int inChannels,
        int height,
        int width,
        int outChannels,
        int padH,
        int padW,
        int outputH,
        int outputW)
    {
        int inputChannelSize = height * width;
        int outputChannelSize = outputH * outputW;
        int inputImageSize = inChannels * inputChannelSize;
        int outputImageSize = outChannels * outputChannelSize;

        for (int b = 0; b < batch; b++)
        {
            int inputBatchOffset = b * inputImageSize;
            int outputBatchOffset = b * outputImageSize;

            for (int oc = 0; oc < outChannels; oc++)
            {
                int outputChannelOffset = outputBatchOffset + oc * outputChannelSize;
                var outputChannel = output.Slice(outputChannelOffset, outputChannelSize);

                for (int ic = 0; ic < inChannels; ic++)
                {
                    int inputChannelOffset = inputBatchOffset + ic * inputChannelSize;
                    int kernelOffset = (oc * inChannels + ic) * 9;

                    Conv2D3x3SingleChannel(
                        input.Slice(inputChannelOffset, inputChannelSize),
                        kernel.Slice(kernelOffset, 9),
                        outputChannel,
                        height, width, outputH, outputW, padH, padW);
                }
            }
        }
    }

    /// <summary>
    /// Process single input channel -> single output channel contribution for 3x3 kernel.
    /// Uses SIMD vectorization for the inner width loop.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Conv2D3x3SingleChannel(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> kernel,
        Span<float> output,
        int inputH,
        int inputW,
        int outputH,
        int outputW,
        int padH,
        int padW)
    {
        // Extract kernel values
        float k00 = kernel[0], k01 = kernel[1], k02 = kernel[2];
        float k10 = kernel[3], k11 = kernel[4], k12 = kernel[5];
        float k20 = kernel[6], k21 = kernel[7], k22 = kernel[8];

#if NET8_0_OR_GREATER
        if (Avx2.IsSupported && outputW >= 8)
        {
            Conv2D3x3SingleChannelAvx2(input, output, inputH, inputW, outputH, outputW, padH, padW,
                k00, k01, k02, k10, k11, k12, k20, k21, k22);
            return;
        }
#endif

        // Scalar fallback
        Conv2D3x3SingleChannelScalar(input, output, inputH, inputW, outputH, outputW, padH, padW,
            k00, k01, k02, k10, k11, k12, k20, k21, k22);
    }

#if NET8_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Conv2D3x3SingleChannelPtr(
        float* input,
        float* kernel,
        float* output,
        int inputH,
        int inputW,
        int outputH,
        int outputW,
        int padH,
        int padW)
    {
        // Extract kernel values
        float k00 = kernel[0], k01 = kernel[1], k02 = kernel[2];
        float k10 = kernel[3], k11 = kernel[4], k12 = kernel[5];
        float k20 = kernel[6], k21 = kernel[7], k22 = kernel[8];

        if (Avx2.IsSupported && outputW >= 8)
        {
            Conv2D3x3SingleChannelAvx2Ptr(input, output, inputH, inputW, outputH, outputW, padH, padW,
                k00, k01, k02, k10, k11, k12, k20, k21, k22);
            return;
        }

        Conv2D3x3SingleChannelScalarPtr(input, output, inputH, inputW, outputH, outputW, padH, padW,
            k00, k01, k02, k10, k11, k12, k20, k21, k22);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Conv2D3x3SingleChannelAvx2Ptr(
        float* input,
        float* output,
        int inputH,
        int inputW,
        int outputH,
        int outputW,
        int padH,
        int padW,
        float k00, float k01, float k02,
        float k10, float k11, float k12,
        float k20, float k21, float k22)
    {
        var vk00 = Vector256.Create(k00);
        var vk01 = Vector256.Create(k01);
        var vk02 = Vector256.Create(k02);
        var vk10 = Vector256.Create(k10);
        var vk11 = Vector256.Create(k11);
        var vk12 = Vector256.Create(k12);
        var vk20 = Vector256.Create(k20);
        var vk21 = Vector256.Create(k21);
        var vk22 = Vector256.Create(k22);

        int vectorWidth = Vector256<float>.Count; // 8

        for (int oh = 0; oh < outputH; oh++)
        {
            int ih = oh - padH;
            float* outRow = output + oh * outputW;

            // Determine valid input rows
            bool row0Valid = ih >= 0 && ih < inputH;
            bool row1Valid = ih + 1 >= 0 && ih + 1 < inputH;
            bool row2Valid = ih + 2 >= 0 && ih + 2 < inputH;

            float* inRow0 = row0Valid ? input + ih * inputW : null;
            float* inRow1 = row1Valid ? input + (ih + 1) * inputW : null;
            float* inRow2 = row2Valid ? input + (ih + 2) * inputW : null;

            int ow = 0;

            // Process 8 outputs at a time
            for (; ow <= outputW - vectorWidth; ow += vectorWidth)
            {
                int iw = ow - padW;
                var acc = Avx.LoadVector256(outRow + ow);

                // Row 0
                if (row0Valid)
                {
                    if (iw >= 0 && iw + vectorWidth + 2 <= inputW)
                    {
                        // All inputs valid - fast path
                        var v0 = Avx.LoadVector256(inRow0 + iw);
                        var v1 = Avx.LoadVector256(inRow0 + iw + 1);
                        var v2 = Avx.LoadVector256(inRow0 + iw + 2);
                        acc = Avx.Add(acc, Avx.Multiply(v0, vk00));
                        acc = Avx.Add(acc, Avx.Multiply(v1, vk01));
                        acc = Avx.Add(acc, Avx.Multiply(v2, vk02));
                    }
                    else
                    {
                        // Boundary handling
                        acc = AddRow3x3Boundary(acc, inRow0, inputW, iw, vk00, vk01, vk02);
                    }
                }

                // Row 1
                if (row1Valid)
                {
                    if (iw >= 0 && iw + vectorWidth + 2 <= inputW)
                    {
                        var v0 = Avx.LoadVector256(inRow1 + iw);
                        var v1 = Avx.LoadVector256(inRow1 + iw + 1);
                        var v2 = Avx.LoadVector256(inRow1 + iw + 2);
                        acc = Avx.Add(acc, Avx.Multiply(v0, vk10));
                        acc = Avx.Add(acc, Avx.Multiply(v1, vk11));
                        acc = Avx.Add(acc, Avx.Multiply(v2, vk12));
                    }
                    else
                    {
                        acc = AddRow3x3Boundary(acc, inRow1, inputW, iw, vk10, vk11, vk12);
                    }
                }

                // Row 2
                if (row2Valid)
                {
                    if (iw >= 0 && iw + vectorWidth + 2 <= inputW)
                    {
                        var v0 = Avx.LoadVector256(inRow2 + iw);
                        var v1 = Avx.LoadVector256(inRow2 + iw + 1);
                        var v2 = Avx.LoadVector256(inRow2 + iw + 2);
                        acc = Avx.Add(acc, Avx.Multiply(v0, vk20));
                        acc = Avx.Add(acc, Avx.Multiply(v1, vk21));
                        acc = Avx.Add(acc, Avx.Multiply(v2, vk22));
                    }
                    else
                    {
                        acc = AddRow3x3Boundary(acc, inRow2, inputW, iw, vk20, vk21, vk22);
                    }
                }

                Avx.Store(outRow + ow, acc);
            }

            // Scalar tail
            for (; ow < outputW; ow++)
            {
                int iw = ow - padW;
                float sum = outRow[ow];

                sum += GetSafe(inRow0, inputW, iw) * k00;
                sum += GetSafe(inRow0, inputW, iw + 1) * k01;
                sum += GetSafe(inRow0, inputW, iw + 2) * k02;
                sum += GetSafe(inRow1, inputW, iw) * k10;
                sum += GetSafe(inRow1, inputW, iw + 1) * k11;
                sum += GetSafe(inRow1, inputW, iw + 2) * k12;
                sum += GetSafe(inRow2, inputW, iw) * k20;
                sum += GetSafe(inRow2, inputW, iw + 1) * k21;
                sum += GetSafe(inRow2, inputW, iw + 2) * k22;

                outRow[ow] = sum;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<float> AddRow3x3Boundary(
        Vector256<float> acc,
        float* row,
        int rowWidth,
        int iw,
        Vector256<float> k0,
        Vector256<float> k1,
        Vector256<float> k2)
    {
        if (row == null) return acc;

        // Load with boundary checks
        Span<float> v0 = stackalloc float[8];
        Span<float> v1 = stackalloc float[8];
        Span<float> v2 = stackalloc float[8];

        for (int i = 0; i < 8; i++)
        {
            int idx0 = iw + i;
            int idx1 = iw + i + 1;
            int idx2 = iw + i + 2;
            v0[i] = (idx0 >= 0 && idx0 < rowWidth) ? row[idx0] : 0f;
            v1[i] = (idx1 >= 0 && idx1 < rowWidth) ? row[idx1] : 0f;
            v2[i] = (idx2 >= 0 && idx2 < rowWidth) ? row[idx2] : 0f;
        }

        fixed (float* pv0 = v0)
        fixed (float* pv1 = v1)
        fixed (float* pv2 = v2)
        {
            var vec0 = Avx.LoadVector256(pv0);
            var vec1 = Avx.LoadVector256(pv1);
            var vec2 = Avx.LoadVector256(pv2);
            acc = Avx.Add(acc, Avx.Multiply(vec0, k0));
            acc = Avx.Add(acc, Avx.Multiply(vec1, k1));
            acc = Avx.Add(acc, Avx.Multiply(vec2, k2));
        }

        return acc;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float GetSafe(float* row, int width, int idx)
    {
        if (row == null || idx < 0 || idx >= width) return 0f;
        return row[idx];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Conv2D3x3SingleChannelScalarPtr(
        float* input,
        float* output,
        int inputH,
        int inputW,
        int outputH,
        int outputW,
        int padH,
        int padW,
        float k00, float k01, float k02,
        float k10, float k11, float k12,
        float k20, float k21, float k22)
    {
        for (int oh = 0; oh < outputH; oh++)
        {
            for (int ow = 0; ow < outputW; ow++)
            {
                float sum = output[oh * outputW + ow];

                for (int kh = 0; kh < 3; kh++)
                {
                    int ih = oh + kh - padH;
                    if (ih < 0 || ih >= inputH) continue;

                    float* inRow = input + ih * inputW;
                    float k0 = kh == 0 ? k00 : (kh == 1 ? k10 : k20);
                    float k1 = kh == 0 ? k01 : (kh == 1 ? k11 : k21);
                    float k2 = kh == 0 ? k02 : (kh == 1 ? k12 : k22);

                    for (int kw = 0; kw < 3; kw++)
                    {
                        int iw = ow + kw - padW;
                        if (iw < 0 || iw >= inputW) continue;

                        float kVal = kw == 0 ? k0 : (kw == 1 ? k1 : k2);
                        sum += inRow[iw] * kVal;
                    }
                }

                output[oh * outputW + ow] = sum;
            }
        }
    }
#endif

    private static void Conv2D3x3SingleChannelScalar(
        ReadOnlySpan<float> input,
        Span<float> output,
        int inputH,
        int inputW,
        int outputH,
        int outputW,
        int padH,
        int padW,
        float k00, float k01, float k02,
        float k10, float k11, float k12,
        float k20, float k21, float k22)
    {
        for (int oh = 0; oh < outputH; oh++)
        {
            for (int ow = 0; ow < outputW; ow++)
            {
                int outIdx = oh * outputW + ow;
                float sum = output[outIdx];

                for (int kh = 0; kh < 3; kh++)
                {
                    int ih = oh + kh - padH;
                    if (ih < 0 || ih >= inputH) continue;

                    int inRowOffset = ih * inputW;

                    for (int kw = 0; kw < 3; kw++)
                    {
                        int iw = ow + kw - padW;
                        if (iw < 0 || iw >= inputW) continue;

                        float kVal = (kh, kw) switch
                        {
                            (0, 0) => k00, (0, 1) => k01, (0, 2) => k02,
                            (1, 0) => k10, (1, 1) => k11, (1, 2) => k12,
                            (2, 0) => k20, (2, 1) => k21, (2, 2) => k22,
                            _ => 0f
                        };
                        sum += input[inRowOffset + iw] * kVal;
                    }
                }

                output[outIdx] = sum;
            }
        }
    }

#if NET8_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Conv2D3x3SingleChannelAvx2(
        ReadOnlySpan<float> input,
        Span<float> output,
        int inputH,
        int inputW,
        int outputH,
        int outputW,
        int padH,
        int padW,
        float k00, float k01, float k02,
        float k10, float k11, float k12,
        float k20, float k21, float k22)
    {
        unsafe
        {
            fixed (float* inputPtr = input)
            fixed (float* outputPtr = output)
            {
                Conv2D3x3SingleChannelAvx2Ptr(inputPtr, outputPtr, inputH, inputW, outputH, outputW, padH, padW,
                    k00, k01, k02, k10, k11, k12, k20, k21, k22);
            }
        }
    }
#endif
}
