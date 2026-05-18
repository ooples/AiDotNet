using System.Buffers;
using AiDotNet.Tensors.Engines.Simd;

namespace AiDotNet.Inference.Quantization;

/// <summary>
/// Per-output-row INT8 weight-only matmul accelerated via the AiDotNet.Tensors
/// SIMD GEMM. Replaces the scalar dequant-on-fly inner loop used by
/// <see cref="QuantizedDenseLayer"/> and <see cref="QuantizedAttentionLayer"/>.
///
/// <para>The shape contract matches both consumers' existing layout:
///   C[m, n] = bias[n] + A[m, k] · dequant(B_int8[n, k], rowScales[n])
/// where dequant(B[r, c], rowScales[r]) = B[r, c] * rowScales[r]. B is stored
/// row-major as [n, k] (one int8 row per output column, the same convention
/// <see cref="Int8WeightOnlyQuantization.QuantizePerRow(Tensor{float})"/>
/// produces).</para>
///
/// <para>Implementation strategy: tile the output dimension so that each
/// dequant-tile of B fits in L2 (≤192 KB by default), call the
/// AVX2-vectorized <see cref="Int8Quantizer.DequantizeInt8ToFloat32"/> per row
/// using that row's scale, then dispatch the FP32 tile through the public
/// <see cref="SimdGemm.Sgemm"/> kernel (transposed B). This composes existing
/// engine-level SIMD primitives — no System.Numerics, no scalar inner loop.
/// Weights stay int8 in DRAM; only the active tile is materialized in FP32.</para>
/// </summary>
internal static class Int8WeightOnlyMatMul
{
    private const int TargetTileBytes = 192 * 1024;
    private const int MinOutputTile = 16;

    /// <summary>
    /// Choose an output-row tile size that keeps the dequant scratch in L2 and
    /// stays a multiple of 16 (matches the BLIS micro-kernel's <c>Nr</c>).
    /// </summary>
    internal static int ChooseTileSize(int outputSize, int inputSize)
    {
        if (outputSize <= 0 || inputSize <= 0) return outputSize;
        long rowBytes = (long)inputSize * sizeof(float);
        if (rowBytes <= 0) return outputSize;
        int candidate = (int)(TargetTileBytes / rowBytes);
        if (candidate < MinOutputTile) candidate = MinOutputTile;
        candidate = candidate / 16 * 16;
        if (candidate < MinOutputTile) candidate = MinOutputTile;
        if (candidate > outputSize) candidate = outputSize;
        return candidate;
    }

    /// <summary>
    /// Compute <c>output[r, o] = (biases?[o] ?? 0) + sum_i input[r, i] * weightsInt8[o, i] * rowScales[o]</c>
    /// for all <c>r in [0, rows)</c> and <c>o in [0, outputSize)</c>. Uses
    /// AiDotNet.Tensors' tiled SGEMM + AVX2 INT8 dequantizer; correctness path
    /// when AVX is unavailable is the engine's scalar fallback (still vector-
    /// register friendly via the BLIS micro-kernel).
    /// </summary>
    public static void MultiplyAddBias(
        ReadOnlySpan<float> input,
        sbyte[] weightsInt8,
        float[] rowScales,
        float[]? biases,
        Span<float> output,
        int rows,
        int inputSize,
        int outputSize)
    {
        if (weightsInt8.Length < (long)outputSize * inputSize)
            throw new ArgumentException("weightsInt8 too small for [outputSize, inputSize].", nameof(weightsInt8));
        if (rowScales.Length < outputSize)
            throw new ArgumentException("rowScales must have outputSize entries.", nameof(rowScales));
        if (biases != null && biases.Length < outputSize)
            throw new ArgumentException("biases (when non-null) must have outputSize entries.", nameof(biases));
        if (input.Length < (long)rows * inputSize)
            throw new ArgumentException("input span too small for [rows, inputSize].", nameof(input));
        if (output.Length < (long)rows * outputSize)
            throw new ArgumentException("output span too small for [rows, outputSize].", nameof(output));
        if (rows == 0 || outputSize == 0)
        {
            output.Clear();
            return;
        }

        int outputTile = ChooseTileSize(outputSize, inputSize);

        var pool = ArrayPool<float>.Shared;
        float[] dequantScratch = pool.Rent(outputTile * inputSize);
        float[] tileOutput = pool.Rent(rows * outputTile);

        try
        {
            for (int oBase = 0; oBase < outputSize; oBase += outputTile)
            {
                int tileN = Math.Min(outputTile, outputSize - oBase);
                int dequantLen = tileN * inputSize;
                int tileOutLen = rows * tileN;

                // Dequantize tileN consecutive rows of int8 weights with per-row
                // scale into the scratch buffer. Each call uses AVX2 internally
                // when supported (see Int8Quantizer.DequantizeInt8ToFloat32).
                for (int oo = 0; oo < tileN; oo++)
                {
                    int o = oBase + oo;
                    int srcStart = o * inputSize;
                    int dstStart = oo * inputSize;
                    Int8Quantizer.DequantizeInt8ToFloat32(
                        new ReadOnlySpan<sbyte>(weightsInt8, srcStart, inputSize),
                        dequantScratch.AsSpan(dstStart, inputSize),
                        rowScales[o]);
                }

                // C_tile [rows, tileN] = A [rows, inputSize] @ B_tile^T,
                // where B_tile [tileN, inputSize] row-major is the logical
                // transpose of the matmul operand [inputSize, tileN].
                SimdGemm.Sgemm(
                    a: input,
                    lda: inputSize,
                    transA: false,
                    b: new ReadOnlySpan<float>(dequantScratch, 0, dequantLen),
                    ldb: inputSize,
                    transB: true,
                    c: tileOutput.AsSpan(0, tileOutLen),
                    m: rows,
                    k: inputSize,
                    n: tileN);

                // Scatter tile into the strided output [rows, outputSize],
                // adding the bias for this output-column block.
                for (int r = 0; r < rows; r++)
                {
                    int srcRow = r * tileN;
                    int dstRow = r * outputSize + oBase;
                    if (biases != null)
                    {
                        for (int oo = 0; oo < tileN; oo++)
                            output[dstRow + oo] = tileOutput[srcRow + oo] + biases[oBase + oo];
                    }
                    else
                    {
                        var src = tileOutput.AsSpan(srcRow, tileN);
                        src.CopyTo(output.Slice(dstRow, tileN));
                    }
                }
            }
        }
        finally
        {
            pool.Return(dequantScratch);
            pool.Return(tileOutput);
        }
    }
}
