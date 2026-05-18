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
    internal static void MultiplyAddBias(
        ReadOnlySpan<float> input,
        sbyte[] weightsInt8,
        float[] rowScales,
        float[]? biases,
        Span<float> output,
        int rows,
        int inputSize,
        int outputSize)
    {
        if (rows < 0)
            throw new ArgumentOutOfRangeException(nameof(rows), rows, "rows must be non-negative.");
        if (inputSize < 0)
            throw new ArgumentOutOfRangeException(nameof(inputSize), inputSize, "inputSize must be non-negative.");
        if (outputSize < 0)
            throw new ArgumentOutOfRangeException(nameof(outputSize), outputSize, "outputSize must be non-negative.");

        long expectedWeights = (long)outputSize * inputSize;
        if (weightsInt8.Length < expectedWeights)
            throw new ArgumentException(
                $"weightsInt8 too small for [outputSize={outputSize}, inputSize={inputSize}]: " +
                $"expected at least {expectedWeights} entries, got {weightsInt8.Length}.",
                nameof(weightsInt8));
        if (rowScales.Length < outputSize)
            throw new ArgumentException(
                $"rowScales must have at least outputSize={outputSize} entries, got {rowScales.Length}.",
                nameof(rowScales));
        if (biases != null && biases.Length < outputSize)
            throw new ArgumentException(
                $"biases (when non-null) must have at least outputSize={outputSize} entries, got {biases.Length}.",
                nameof(biases));

        long expectedInput = (long)rows * inputSize;
        if (input.Length < expectedInput)
            throw new ArgumentException(
                $"input span too small for [rows={rows}, inputSize={inputSize}]: " +
                $"expected at least {expectedInput} entries, got {input.Length}.",
                nameof(input));

        long expectedOutput = (long)rows * outputSize;
        if (output.Length < expectedOutput)
            throw new ArgumentException(
                $"output span too small for [rows={rows}, outputSize={outputSize}]: " +
                $"expected at least {expectedOutput} entries, got {output.Length}.",
                nameof(output));

        // No-op when the logical output is empty. Do NOT clear `output` here —
        // the caller's span may be larger than the logical [rows, outputSize]
        // region and `output.Clear()` would zero stale-but-valid data outside
        // the region this call is responsible for.
        if (rows == 0 || outputSize == 0)
            return;

        // Reject inputSize == 0 explicitly. The two production callers
        // (QuantizedDenseLayer, QuantizedAttentionLayer Q/K/V/O projections)
        // validate input dim upstream; an inputSize of 0 only arrives via a
        // misconfigured caller, never via legitimate workloads. Failing
        // fast surfaces the misconfiguration at the matmul call instead of
        // producing a silent zero-matrix result that would mask the upstream
        // bug (review #1363 C6XGR: prior permissive bias-or-zero return
        // turned this edge case into a silent no-op).
        if (inputSize == 0)
            throw new ArgumentOutOfRangeException(
                nameof(inputSize), 0,
                "inputSize must be positive when rows > 0 and outputSize > 0; " +
                "the empty-output early-return covers the rows==0 / outputSize==0 cases.");

        int outputTile = ChooseTileSize(outputSize, inputSize);

        // Allocate using long-typed sizes so overflow against int.MaxValue
        // surfaces explicitly (e.g. outputTile=3000 × inputSize=4096 +
        // future scale-up to 1M-row batches would silently wrap to a
        // negative int and pass to Rent — review #1363 C6XFg). At the
        // canary shapes covered by tests these products stay well under
        // 2 GiB / 4 bytes per float, but the bound check matters once
        // wider FFN / longer sequences land.
        long dequantScratchLen = (long)outputTile * inputSize;
        long tileOutputLen = (long)rows * outputTile;
        if (dequantScratchLen > int.MaxValue || tileOutputLen > int.MaxValue)
            throw new InvalidOperationException(
                $"Tiled INT8 matmul exceeded int.MaxValue per-tile buffer ({dequantScratchLen}, {tileOutputLen}). " +
                "Reduce outputTile or split rows externally before calling MultiplyAddBias.");

        var pool = ArrayPool<float>.Shared;
        // Both rents inside the try block so that if the SECOND rent
        // throws (rare: pool exhaustion / OOM), the first buffer is
        // still returned to the pool by the finally (review #1363
        // C6XF6 — prior code rented before try and leaked dequantScratch
        // on a tileOutput rent throw).
        float[]? dequantScratch = null;
        float[]? tileOutput = null;
        try
        {
            dequantScratch = pool.Rent((int)dequantScratchLen);
            tileOutput = pool.Rent((int)tileOutputLen);
            for (int oBase = 0; oBase < outputSize; oBase += outputTile)
            {
                int tileN = Math.Min(outputTile, outputSize - oBase);
                int dequantLen = tileN * inputSize;
                int tileOutLen = rows * tileN;

                // Dequantize tileN consecutive rows of int8 weights with per-row
                // scale into the scratch buffer. Each call uses AVX2 internally
                // when supported (see Int8Quantizer.DequantizeInt8ToFloat32) and
                // fully writes its destination row from the int8 source, so the
                // ArrayPool-returned uninitialized scratch is safe even though
                // ArrayPool<T>.Shared.Rent does not zero on rent.
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

                // C_tile [rows, tileN] = A [rows, inputSize] @ B_tile^T (Sgemm
                // overload with transB=true). Explicit tileSpan.Clear() is
                // belt-and-braces against a future SimdGemm.Sgemm refactor
                // that drops its implicit clear of c (review #1363 C6XGz
                // shortened from the 8-line line-number reference).
                var tileSpan = tileOutput.AsSpan(0, tileOutLen);
                tileSpan.Clear();
                SimdGemm.Sgemm(
                    a: input,
                    lda: inputSize,
                    transA: false,
                    b: new ReadOnlySpan<float>(dequantScratch, 0, dequantLen),
                    ldb: inputSize,
                    transB: true,
                    c: tileSpan,
                    m: rows,
                    k: inputSize,
                    n: tileN);

                // Scatter tile into the strided output [rows, outputSize],
                // adding the bias for this output-column block via
                // AiDotNet.Tensors' SimdKernels.VectorAdd — the same
                // accelerated SIMD primitive SimdGemm uses. Per the
                // project-level rule, AiDotNet.Tensors is the in-tree
                // SIMD library (full PyTorch parity); System.Numerics is
                // banned. VectorAdd is one call per output row; tileN is
                // a multiple of 16 per ChooseTileSize's alignment
                // contract, keeping the kernel on its best-case path
                // (review #1363 C6XGD).
                for (int r = 0; r < rows; r++)
                {
                    int srcRow = r * tileN;
                    int dstRow = r * outputSize + oBase;
                    var srcSpan = tileOutput.AsSpan(srcRow, tileN);
                    var dstSpan = output.Slice(dstRow, tileN);
                    if (biases != null)
                    {
                        var biasSpan = new ReadOnlySpan<float>(biases, oBase, tileN);
                        AiDotNet.Tensors.Engines.Simd.SimdKernels.VectorAdd(srcSpan, biasSpan, dstSpan);
                    }
                    else
                    {
                        srcSpan.CopyTo(dstSpan);
                    }
                }
            }
        }
        finally
        {
            // Null-conditional Return so a mid-rent throw (only the FIRST
            // rent succeeded) still returns what we have without an NRE.
            if (dequantScratch is not null) pool.Return(dequantScratch);
            if (tileOutput is not null) pool.Return(tileOutput);
        }
    }
}
