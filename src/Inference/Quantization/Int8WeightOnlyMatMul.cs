using System.Buffers;
using AiDotNet.Tensors.Engines.Simd;

namespace AiDotNet.Inference.Quantization;

/// <summary>
/// Per-output-row INT8 weight-only matmul. The shape contract:
///   C[m, n] = bias[n] + A[m, k] · dequant(B_int8[n, k], rowScales[n])
/// where dequant(B[r, c], rowScales[r]) = B[r, c] * rowScales[r]. B is stored
/// row-major as [n, k] (one int8 row per output column, the same convention
/// <see cref="Int8WeightOnlyQuantization.QuantizePerRow(Tensor{float})"/>
/// produces).
///
/// <para>Implementation: routes through
/// <see cref="SimdGemm.SgemmWithInt8RowScaledCachedB"/> — a tiled GEMM that
/// keeps weights in INT8 all the way through the macro-kernel and folds the
/// per-row scales into the per-tile dequant. The pre-packed cache is keyed
/// on the <c>sbyte[]</c> reference and survives across <c>Predict</c> calls,
/// so the per-call cost reduces to PackA (activations only) + macro-kernel
/// dispatch. INT8 weights stay 4× smaller than FP32 in DRAM, finally
/// realizing the bandwidth saving that motivated the quantization in the
/// first place. Closes AiDotNet#1349 once the Tensors NuGet ships with
/// <see cref="SimdGemm.SgemmWithInt8RowScaledCachedB"/> (Tensors PR #427 /
/// issue ooples/AiDotNet.Tensors#401).</para>
///
/// <para>Pre-#1349-fix path was: per-call ArrayPool rent + AVX2 dequant per
/// tile into FP32 scratch + standard FP32 Sgemm + bias-scatter. That
/// defeated the INT8 memory-bandwidth advantage because the FP32 scratch
/// was the size of the active weight tile (and grew with batch). The new
/// path never materializes the full FP32 weights — dequant is L1-resident
/// inside the macro-kernel.</para>
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
                "the empty-output early-return covers the rows==0 / outputSize==0 cases. " +
                "Note: this is a BEHAVIOR CHANGE from the prior #1348 surface, which silently " +
                "produced a zero-or-bias-only output for inputSize==0. Callers that relied on " +
                "that fall-through must now either guard the call themselves or pre-fill the " +
                "output span with the bias values before invocation (review #1363 C8QXn).");

        // Single tiled INT8 GEMM call — weights stay int8 inside the cache
        // and the macro-kernel's per-tile dequant, never materializing the
        // full FP32 weight matrix. ChooseTileSize / per-call ArrayPool /
        // outer dequant-then-Sgemm pattern were the source of the ~20×
        // wall-clock gap reported in #1349; the new kernel collapses all
        // of that to one call.
        SimdGemm.SgemmWithInt8RowScaledCachedB(
            a: input,
            bInt8: weightsInt8,
            rowScales: new ReadOnlySpan<float>(rowScales, 0, outputSize),
            c: output.Slice(0, rows * outputSize),
            m: rows,
            k: inputSize,
            n: outputSize);

        // Bias-add. The new kernel writes C without bias; fold biases in
        // via the same SimdKernels.VectorAdd primitive used before — one
        // call per output row, AVX2-accelerated when available, scalar
        // epilogue for the unaligned tail.
        if (biases != null)
        {
            for (int r = 0; r < rows; r++)
            {
                int dstRow = r * outputSize;
                var rowSpan = output.Slice(dstRow, outputSize);
                AiDotNet.Tensors.Engines.Simd.SimdKernels.VectorAdd(
                    rowSpan, new ReadOnlySpan<float>(biases, 0, outputSize), rowSpan);
            }
        }
    }
}
