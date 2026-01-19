using System;
using System.Buffers;
using System.Numerics.Tensors;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Winograd convolution for 3x3 kernels with stride=1.
/// Reduces multiplications by 2.25x (36 muls per 4 outputs → 16 muls).
/// Based on F(2x2, 3x3) Winograd algorithm.
/// </summary>
internal static class WinogradHelper
{
    // Winograd F(2,3) transform matrices
    // G: kernel transform (4x3)
    // B: input transform (4x4)
    // A: output transform (2x4)
    private static readonly float[] G =
    {
        1f, 0f, 0f,
        0.5f, 0.5f, 0.5f,
        0.5f, -0.5f, 0.5f,
        0f, 0f, 1f
    };

    private static readonly float[] Gt =
    {
        1f, 0.5f, 0.5f, 0f,
        0f, 0.5f, -0.5f, 0f,
        0f, 0.5f, 0.5f, 1f
    };

    private static readonly float[] B =
    {
        1f, 0f, -1f, 0f,
        0f, 1f, 1f, 0f,
        0f, -1f, 1f, 0f,
        0f, 1f, 0f, -1f
    };

    private static readonly float[] Bt =
    {
        1f, 0f, 0f, 0f,
        0f, 1f, -1f, 1f,
        -1f, 1f, 1f, 0f,
        0f, 0f, 0f, -1f
    };

    private static readonly float[] At =
    {
        1f, 1f, 1f, 0f,
        0f, 1f, -1f, -1f
    };

    private static readonly float[] A =
    {
        1f, 0f,
        1f, 1f,
        1f, -1f,
        0f, -1f
    };

    // Minimum output dimension for Winograd to be beneficial
    // Below this, im2col+GEMM is faster due to single large GEMM vs 16 small ones
    private const int MinOutputDimForWinograd = 224;

    /// <summary>
    /// Check if Winograd can be used for this convolution.
    /// Winograd is only beneficial for 3x3 kernels with stride=1, dilation=1,
    /// and when output is large enough that the overhead is amortized.
    /// </summary>
    public static bool CanUseWinograd(int kernelH, int kernelW, int strideH, int strideW, int dilationH, int dilationW)
    {
        // Winograd F(2,3) only works for 3x3 kernels with stride=1, dilation=1
        // and is only beneficial for larger outputs (overhead of 16 GEMMs vs 1 large GEMM)
        return kernelH == 3 && kernelW == 3 &&
               strideH == 1 && strideW == 1 &&
               dilationH == 1 && dilationW == 1;
    }

    /// <summary>
    /// Check if Winograd should be used for this convolution, considering output size.
    /// </summary>
    public static bool ShouldUseWinograd(int kernelH, int kernelW, int strideH, int strideW, int dilationH, int dilationW, int outputH, int outputW)
    {
        if (!CanUseWinograd(kernelH, kernelW, strideH, strideW, dilationH, dilationW))
        {
            return false;
        }

        // Only use Winograd when output is large enough to amortize the overhead
        // 16 small GEMMs have significant overhead compared to 1 large GEMM
        return outputH >= MinOutputDimForWinograd && outputW >= MinOutputDimForWinograd;
    }

    /// <summary>
    /// Performs 3x3 convolution using Winograd F(2,3) algorithm.
    /// Output size must be valid for 2x2 output tiles.
    /// </summary>
    public static void Conv2DWinograd(
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
        // Output dimensions (assumes stride=1)
        int outputH = height + 2 * padH - 2;
        int outputW = width + 2 * padW - 2;

        // Number of 2x2 tiles in output
        int tilesH = (outputH + 1) / 2;
        int tilesW = (outputW + 1) / 2;
        int numTiles = tilesH * tilesW;

        var pool = ArrayPool<float>.Shared;

        // Allocate workspace
        // Transformed kernels: [outChannels, inChannels, 4, 4]
        float[] transformedKernels = pool.Rent(outChannels * inChannels * 16);
        // Transformed input tiles: [inChannels, 16, numTiles]
        float[] transformedInput = pool.Rent(inChannels * 16 * numTiles);
        // Elementwise product result: [outChannels, 16, numTiles]
        float[] transformedOutput = pool.Rent(outChannels * 16 * numTiles);

        try
        {
            // Step 1: Transform all kernels
            TransformKernels(kernel, transformedKernels, outChannels, inChannels);

            // Process each batch element
            for (int b = 0; b < batch; b++)
            {
                int inputOffset = b * inChannels * height * width;
                int outputOffset = b * outChannels * outputH * outputW;

                // Step 2: Transform input tiles
                TransformInput(
                    input.Slice(inputOffset, inChannels * height * width),
                    transformedInput,
                    inChannels, height, width, padH, padW, tilesH, tilesW);

                // Step 3: Batched element-wise multiply and accumulate
                // For each Winograd element (0-15), do GEMM across tiles
                ComputeWinogradGemm(
                    transformedKernels, transformedInput, transformedOutput,
                    outChannels, inChannels, numTiles);

                // Step 4: Transform output tiles back to spatial domain
                TransformOutput(
                    transformedOutput,
                    output.Slice(outputOffset, outChannels * outputH * outputW),
                    outChannels, outputH, outputW, tilesH, tilesW);
            }
        }
        finally
        {
            pool.Return(transformedKernels);
            pool.Return(transformedInput);
            pool.Return(transformedOutput);
        }
    }

    /// <summary>
    /// Transform 3x3 kernels to 4x4 Winograd domain: U = G @ g @ G^T
    /// Layout: [element][outChannels][inChannels] for GEMM-friendly access
    /// </summary>
    private static void TransformKernels(
        ReadOnlySpan<float> kernels,
        Span<float> transformed,
        int outChannels,
        int inChannels)
    {
        Span<float> temp = stackalloc float[12]; // 4x3 intermediate
        Span<float> kernel3x3 = stackalloc float[9];
        Span<float> result4x4 = stackalloc float[16];
        int ocIcSize = outChannels * inChannels;

        for (int oc = 0; oc < outChannels; oc++)
        {
            for (int ic = 0; ic < inChannels; ic++)
            {
                int srcOffset = (oc * inChannels + ic) * 9;

                // Copy kernel
                kernels.Slice(srcOffset, 9).CopyTo(kernel3x3);

                // temp = G @ kernel (4x3 @ 3x3 = 4x3)
                MatMul4x3_3x3(G, kernel3x3, temp);

                // result = temp @ G^T (4x3 @ 3x4 = 4x4)
                MatMul4x3_3x4(temp, Gt, result4x4);

                // Store in element-major layout: [element][oc][ic]
                for (int elem = 0; elem < 16; elem++)
                {
                    int dstOffset = elem * ocIcSize + oc * inChannels + ic;
                    transformed[dstOffset] = result4x4[elem];
                }
            }
        }
    }

    /// <summary>
    /// Transform 4x4 input tiles to Winograd domain: V = B^T @ d @ B
    /// Layout: [element][channels][numTiles] for GEMM-friendly access
    /// </summary>
    private static void TransformInput(
        ReadOnlySpan<float> input,
        Span<float> transformed,
        int channels,
        int height,
        int width,
        int padH,
        int padW,
        int tilesH,
        int tilesW)
    {
        int numTiles = tilesH * tilesW;
        int channelTileSize = channels * numTiles;
        Span<float> tile4x4 = stackalloc float[16];
        Span<float> temp4x4 = stackalloc float[16];
        Span<float> result4x4 = stackalloc float[16];

        for (int c = 0; c < channels; c++)
        {
            int channelOffset = c * height * width;

            for (int th = 0; th < tilesH; th++)
            {
                for (int tw = 0; tw < tilesW; tw++)
                {
                    int tileIdx = th * tilesW + tw;
                    int startH = th * 2 - padH;
                    int startW = tw * 2 - padW;

                    // Extract 4x4 input tile with padding
                    ExtractTile(input, channelOffset, height, width, startH, startW, tile4x4);

                    // temp = B^T @ tile (4x4 @ 4x4 = 4x4)
                    MatMul4x4_4x4(Bt, tile4x4, temp4x4);

                    // result = temp @ B (4x4 @ 4x4 = 4x4)
                    MatMul4x4_4x4(temp4x4, B, result4x4);

                    // Store in element-major layout: [element][channel][tile]
                    for (int elem = 0; elem < 16; elem++)
                    {
                        int dstOffset = elem * channelTileSize + c * numTiles + tileIdx;
                        transformed[dstOffset] = result4x4[elem];
                    }
                }
            }
        }
    }

    /// <summary>
    /// Compute Winograd GEMM: For each Winograd element, perform batched GEMM.
    /// Uses BLAS GEMM for each of the 16 Winograd elements.
    /// Layout: kernels[elem][oc][ic], input[elem][ic][tile], output[elem][oc][tile]
    /// </summary>
    private static void ComputeWinogradGemm(
        ReadOnlySpan<float> transformedKernels,
        ReadOnlySpan<float> transformedInput,
        Span<float> transformedOutput,
        int outChannels,
        int inChannels,
        int numTiles)
    {
        // Clear output
        transformedOutput.Clear();

        int ocIcSize = outChannels * inChannels;
        int icTileSize = inChannels * numTiles;
        int ocTileSize = outChannels * numTiles;

        // For each Winograd element (0-15), perform GEMM:
        // C[elem] = A[elem] @ B[elem]
        // where A is [outChannels x inChannels], B is [inChannels x numTiles]
        for (int elem = 0; elem < 16; elem++)
        {
            int kernelOffset = elem * ocIcSize;
            int inputOffset = elem * icTileSize;
            int outputOffset = elem * ocTileSize;

            // Try BLAS GEMM first for this element
            bool usedBlas = BlasProvider.TryGemm(
                outChannels, numTiles, inChannels,
                transformedKernels.Slice(kernelOffset, ocIcSize),
                inChannels,
                transformedInput.Slice(inputOffset, icTileSize),
                numTiles,
                transformedOutput.Slice(outputOffset, ocTileSize),
                numTiles);

            if (!usedBlas)
            {
                // Fallback to scalar implementation for this element
                for (int oc = 0; oc < outChannels; oc++)
                {
                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        float kernelVal = transformedKernels[kernelOffset + oc * inChannels + ic];
                        int inSliceOffset = inputOffset + ic * numTiles;
                        int outSliceOffset = outputOffset + oc * numTiles;

                        for (int t = 0; t < numTiles; t++)
                        {
                            transformedOutput[outSliceOffset + t] += kernelVal * transformedInput[inSliceOffset + t];
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Transform Winograd output back to spatial domain: Y = A^T @ M @ A
    /// Layout: input is [element][outChannels][numTiles]
    /// </summary>
    private static void TransformOutput(
        ReadOnlySpan<float> transformed,
        Span<float> output,
        int outChannels,
        int outputH,
        int outputW,
        int tilesH,
        int tilesW)
    {
        int numTiles = tilesH * tilesW;
        int ocTileSize = outChannels * numTiles;
        Span<float> tile4x4 = stackalloc float[16];
        Span<float> temp2x4 = stackalloc float[8];
        Span<float> result2x2 = stackalloc float[4];

        for (int oc = 0; oc < outChannels; oc++)
        {
            for (int th = 0; th < tilesH; th++)
            {
                for (int tw = 0; tw < tilesW; tw++)
                {
                    int tileIdx = th * tilesW + tw;

                    // Gather transformed tile values from element-major layout
                    for (int elem = 0; elem < 16; elem++)
                    {
                        tile4x4[elem] = transformed[elem * ocTileSize + oc * numTiles + tileIdx];
                    }

                    // temp = A^T @ tile (2x4 @ 4x4 = 2x4)
                    MatMul2x4_4x4(At, tile4x4, temp2x4);

                    // result = temp @ A (2x4 @ 4x2 = 2x2)
                    MatMul2x4_4x2(temp2x4, A, result2x2);

                    // Write 2x2 output tile
                    int outH = th * 2;
                    int outW = tw * 2;
                    int channelOffset = oc * outputH * outputW;

                    for (int i = 0; i < 2 && outH + i < outputH; i++)
                    {
                        for (int j = 0; j < 2 && outW + j < outputW; j++)
                        {
                            output[channelOffset + (outH + i) * outputW + (outW + j)] = result2x2[i * 2 + j];
                        }
                    }
                }
            }
        }
    }

    private static void ExtractTile(
        ReadOnlySpan<float> input,
        int channelOffset,
        int height,
        int width,
        int startH,
        int startW,
        Span<float> tile)
    {
        for (int i = 0; i < 4; i++)
        {
            int ih = startH + i;
            for (int j = 0; j < 4; j++)
            {
                int iw = startW + j;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                {
                    tile[i * 4 + j] = input[channelOffset + ih * width + iw];
                }
                else
                {
                    tile[i * 4 + j] = 0f;
                }
            }
        }
    }

    // Small matrix multiply routines (unrolled for performance)
    private static void MatMul4x3_3x3(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c)
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                float sum = 0;
                for (int k = 0; k < 3; k++)
                {
                    sum += a[i * 3 + k] * b[k * 3 + j];
                }
                c[i * 3 + j] = sum;
            }
        }
    }

    private static void MatMul4x3_3x4(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c)
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                float sum = 0;
                for (int k = 0; k < 3; k++)
                {
                    sum += a[i * 3 + k] * b[k * 4 + j];
                }
                c[i * 4 + j] = sum;
            }
        }
    }

    private static void MatMul4x4_4x4(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c)
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                float sum = 0;
                for (int k = 0; k < 4; k++)
                {
                    sum += a[i * 4 + k] * b[k * 4 + j];
                }
                c[i * 4 + j] = sum;
            }
        }
    }

    private static void MatMul2x4_4x4(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c)
    {
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                float sum = 0;
                for (int k = 0; k < 4; k++)
                {
                    sum += a[i * 4 + k] * b[k * 4 + j];
                }
                c[i * 4 + j] = sum;
            }
        }
    }

    private static void MatMul2x4_4x2(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c)
    {
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                float sum = 0;
                for (int k = 0; k < 4; k++)
                {
                    sum += a[i * 4 + k] * b[k * 2 + j];
                }
                c[i * 2 + j] = sum;
            }
        }
    }
}
