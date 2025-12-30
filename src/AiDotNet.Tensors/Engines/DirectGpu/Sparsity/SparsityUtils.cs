// Copyright (c) AiDotNet. All rights reserved.
// Sparsity utilities for 2:4 structured sparsity pattern detection and enforcement.

using System;
using System.Numerics;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.Sparsity;

/// <summary>
/// Utilities for working with 2:4 structured sparsity.
/// </summary>
/// <remarks>
/// <para><b>2:4 Structured Sparsity:</b></para>
/// <para>
/// A sparsity pattern where every group of 4 consecutive elements contains exactly 2 zeros.
/// This provides predictable memory access patterns and enables hardware acceleration
/// on AMD MI200+ and NVIDIA Ampere+ GPUs.
/// </para>
/// <para><b>Benefits:</b></para>
/// <list type="bullet">
/// <item>2x memory compression</item>
/// <item>2x compute reduction</item>
/// <item>Predictable memory access patterns</item>
/// <item>Hardware-accelerated sparse tensor cores</item>
/// </list>
/// </remarks>
public static class SparsityUtils
{
    /// <summary>
    /// Checks if a matrix already has a valid 2:4 sparsity pattern.
    /// </summary>
    /// <param name="data">Matrix data in row-major order.</param>
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns (must be divisible by 4).</param>
    /// <param name="threshold">Values below this threshold are considered zero.</param>
    /// <returns>True if the matrix has valid 2:4 sparsity, false otherwise.</returns>
    public static bool Has2x4SparsityPattern(float[] data, int rows, int cols, float threshold = 1e-6f)
    {
        if (cols % 4 != 0)
            return false;

        int numGroups = cols / 4;

        for (int row = 0; row < rows; row++)
        {
            for (int group = 0; group < numGroups; group++)
            {
                int baseIdx = row * cols + group * 4;
                int zeroCount = 0;

                for (int i = 0; i < 4; i++)
                {
                    if (MathF.Abs(data[baseIdx + i]) < threshold)
                        zeroCount++;
                }

                if (zeroCount != 2)
                    return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Calculates the sparsity ratio of a matrix.
    /// </summary>
    /// <param name="data">Matrix data.</param>
    /// <param name="threshold">Values below this threshold are considered zero.</param>
    /// <returns>Ratio of zero elements (0.0 to 1.0).</returns>
    public static float CalculateSparsityRatio(float[] data, float threshold = 1e-6f)
    {
        if (data.Length == 0)
            return 0f;

        int zeroCount = 0;
        for (int i = 0; i < data.Length; i++)
        {
            if (MathF.Abs(data[i]) < threshold)
                zeroCount++;
        }

        return (float)zeroCount / data.Length;
    }

    /// <summary>
    /// Enforces 2:4 structured sparsity by zeroing the 2 smallest magnitude values in each group of 4.
    /// </summary>
    /// <param name="data">Matrix data to sparsify (modified in place).</param>
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns (must be divisible by 4).</param>
    /// <exception cref="ArgumentException">Thrown if cols is not divisible by 4.</exception>
    public static void Enforce2x4SparsityInPlace(float[] data, int rows, int cols)
    {
        if (cols % 4 != 0)
            throw new ArgumentException("Column count must be divisible by 4 for 2:4 sparsity.", nameof(cols));

        int numGroups = cols / 4;

        for (int row = 0; row < rows; row++)
        {
            for (int group = 0; group < numGroups; group++)
            {
                int baseIdx = row * cols + group * 4;
                EnforceGroupSparsity(data, baseIdx);
            }
        }
    }

    /// <summary>
    /// Enforces 2:4 sparsity on a single group of 4 elements.
    /// </summary>
    private static void EnforceGroupSparsity(float[] data, int baseIdx)
    {
        // Find indices of 2 largest absolute values
        Span<float> absVals = stackalloc float[4];
        Span<int> indices = stackalloc int[] { 0, 1, 2, 3 };

        for (int i = 0; i < 4; i++)
        {
            absVals[i] = MathF.Abs(data[baseIdx + i]);
        }

        // Sort indices by absolute value (descending)
        // Simple bubble sort for 4 elements
        for (int i = 0; i < 3; i++)
        {
            for (int j = i + 1; j < 4; j++)
            {
                if (absVals[indices[j]] > absVals[indices[i]])
                {
                    int tmp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tmp;
                }
            }
        }

        // Zero out the 2 smallest (indices[2] and indices[3])
        data[baseIdx + indices[2]] = 0f;
        data[baseIdx + indices[3]] = 0f;
    }

    /// <summary>
    /// Compresses a 2:4 sparse matrix into the compressed format.
    /// </summary>
    /// <param name="denseData">Dense matrix data with 2:4 sparsity pattern.</param>
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns.</param>
    /// <returns>Compressed sparse representation.</returns>
    public static Compressed2x4Sparse CompressTo2x4(float[] denseData, int rows, int cols)
    {
        if (cols % 4 != 0)
            throw new ArgumentException("Column count must be divisible by 4.", nameof(cols));

        int numGroups = cols / 4;
        int compressedCols = cols / 2;

        var values = new float[rows * compressedCols];
        var indices = new byte[rows * numGroups];

        for (int row = 0; row < rows; row++)
        {
            for (int group = 0; group < numGroups; group++)
            {
                int baseIdx = row * cols + group * 4;

                // Find the two non-zero positions
                int idx0 = -1, idx1 = -1;
                float val0 = 0, val1 = 0;

                for (int i = 0; i < 4; i++)
                {
                    float val = denseData[baseIdx + i];
                    if (MathF.Abs(val) > 1e-10f)
                    {
                        if (idx0 < 0)
                        {
                            idx0 = i;
                            val0 = val;
                        }
                        else
                        {
                            idx1 = i;
                            val1 = val;
                            break;
                        }
                    }
                }

                // Handle edge cases (all zeros or only one non-zero)
                if (idx0 < 0) { idx0 = 0; val0 = 0; }
                if (idx1 < 0) { idx1 = (idx0 + 1) % 4; val1 = 0; }

                // Ensure idx0 < idx1
                if (idx0 > idx1)
                {
                    (idx0, idx1) = (idx1, idx0);
                    (val0, val1) = (val1, val0);
                }

                // Store compressed values
                int valueIdx = row * compressedCols + group * 2;
                values[valueIdx] = val0;
                values[valueIdx + 1] = val1;

                // Pack indices (2 bits each)
                indices[row * numGroups + group] = (byte)((idx1 << 2) | idx0);
            }
        }

        return new Compressed2x4Sparse(values, indices, rows, cols);
    }

    /// <summary>
    /// Decompresses a 2:4 sparse matrix back to dense format.
    /// </summary>
    /// <param name="sparse">Compressed sparse matrix.</param>
    /// <returns>Dense matrix data.</returns>
    public static float[] DecompressFrom2x4(Compressed2x4Sparse sparse)
    {
        int rows = sparse.Rows;
        int cols = sparse.OriginalCols;
        int numGroups = cols / 4;
        int compressedCols = cols / 2;

        var dense = new float[rows * cols];

        for (int row = 0; row < rows; row++)
        {
            for (int group = 0; group < numGroups; group++)
            {
                int baseIdx = row * cols + group * 4;
                byte packed = sparse.Indices[row * numGroups + group];

                int idx0 = packed & 0x3;
                int idx1 = (packed >> 2) & 0x3;

                int valueIdx = row * compressedCols + group * 2;
                float val0 = sparse.Values[valueIdx];
                float val1 = sparse.Values[valueIdx + 1];

                dense[baseIdx + idx0] = val0;
                dense[baseIdx + idx1] = val1;
            }
        }

        return dense;
    }

    /// <summary>
    /// Performs CPU sparse GEMM: C = alpha * A_sparse * B + beta * C
    /// </summary>
    /// <param name="sparseA">Compressed sparse A matrix.</param>
    /// <param name="denseB">Dense B matrix (K x N).</param>
    /// <param name="C">Output C matrix (M x N).</param>
    /// <param name="N">Columns of B and C.</param>
    /// <param name="alpha">Scalar for A*B.</param>
    /// <param name="beta">Scalar for C.</param>
    public static void SparseGemmCpu(
        Compressed2x4Sparse sparseA,
        float[] denseB,
        float[] C,
        int N,
        float alpha = 1.0f,
        float beta = 0.0f)
    {
        int M = sparseA.Rows;
        int K = sparseA.OriginalCols;
        int numGroups = K / 4;
        int compressedK = K / 2;

        // Apply beta to C first
        if (MathF.Abs(beta) < 1e-10f)
        {
            Array.Clear(C, 0, C.Length);
        }
        else if (MathF.Abs(beta - 1.0f) > 1e-10f)
        {
            for (int i = 0; i < C.Length; i++)
                C[i] *= beta;
        }

        // Sparse GEMM
        for (int row = 0; row < M; row++)
        {
            for (int group = 0; group < numGroups; group++)
            {
                int kBase = group * 4;
                byte packed = sparseA.Indices[row * numGroups + group];

                int idx0 = packed & 0x3;
                int idx1 = (packed >> 2) & 0x3;

                int k0 = kBase + idx0;
                int k1 = kBase + idx1;

                int valueIdx = row * compressedK + group * 2;
                float val0 = sparseA.Values[valueIdx];
                float val1 = sparseA.Values[valueIdx + 1];

                // Accumulate contributions to each output column
                for (int col = 0; col < N; col++)
                {
                    int cIdx = row * N + col;
                    C[cIdx] += alpha * (val0 * denseB[k0 * N + col] + val1 * denseB[k1 * N + col]);
                }
            }
        }
    }

    /// <summary>
    /// Estimates the potential speedup from using 2:4 sparsity on a matrix.
    /// </summary>
    /// <param name="data">Matrix data.</param>
    /// <param name="threshold">Zero threshold.</param>
    /// <returns>Estimated speedup factor (1.0 = no benefit, 2.0 = optimal).</returns>
    public static float EstimateSparsityBenefit(float[] data, float threshold = 1e-6f)
    {
        float currentSparsity = CalculateSparsityRatio(data, threshold);

        // 2:4 sparsity gives 50% sparsity = 2x speedup
        // If already sparser, we lose some benefit from having to pad
        // If denser, we lose accuracy from enforcing sparsity

        if (currentSparsity >= 0.5f)
        {
            // Already sparse enough, can achieve full 2x
            return 2.0f;
        }
        else
        {
            // Need to sparsify more, estimate accuracy loss vs speedup tradeoff
            // Linear interpolation: at 0% sparsity, benefit is reduced
            return 1.0f + currentSparsity * 2.0f;
        }
    }
}

/// <summary>
/// Represents a matrix in 2:4 compressed sparse format.
/// </summary>
public sealed class Compressed2x4Sparse
{
    /// <summary>
    /// Non-zero values (M x K/2).
    /// </summary>
    public float[] Values { get; }

    /// <summary>
    /// Packed 2-bit indices indicating positions within each group of 4 (M x K/4).
    /// Each byte contains indices for one group: (idx1 &lt;&lt; 2) | idx0
    /// </summary>
    public byte[] Indices { get; }

    /// <summary>
    /// Number of rows.
    /// </summary>
    public int Rows { get; }

    /// <summary>
    /// Original (uncompressed) number of columns.
    /// </summary>
    public int OriginalCols { get; }

    /// <summary>
    /// Compressed number of columns (OriginalCols / 2).
    /// </summary>
    public int CompressedCols => OriginalCols / 2;

    /// <summary>
    /// Number of groups per row (OriginalCols / 4).
    /// </summary>
    public int GroupsPerRow => OriginalCols / 4;

    /// <summary>
    /// Memory size of compressed representation in bytes.
    /// </summary>
    public long CompressedSizeBytes => Values.Length * sizeof(float) + Indices.Length;

    /// <summary>
    /// Memory size of original dense representation in bytes.
    /// </summary>
    public long OriginalSizeBytes => (long)Rows * OriginalCols * sizeof(float);

    /// <summary>
    /// Compression ratio achieved.
    /// </summary>
    public float CompressionRatio => (float)OriginalSizeBytes / CompressedSizeBytes;

    public Compressed2x4Sparse(float[] values, byte[] indices, int rows, int originalCols)
    {
        Values = values ?? throw new ArgumentNullException(nameof(values));
        Indices = indices ?? throw new ArgumentNullException(nameof(indices));
        Rows = rows;
        OriginalCols = originalCols;

        // Validate sizes
        int expectedValues = rows * (originalCols / 2);
        int expectedIndices = rows * (originalCols / 4);

        if (values.Length != expectedValues)
            throw new ArgumentException($"Values array size mismatch. Expected {expectedValues}, got {values.Length}");
        if (indices.Length != expectedIndices)
            throw new ArgumentException($"Indices array size mismatch. Expected {expectedIndices}, got {indices.Length}");
    }
}
