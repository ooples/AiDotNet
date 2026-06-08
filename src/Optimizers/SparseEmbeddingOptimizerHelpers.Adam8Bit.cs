using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Optimizers;

/// <summary>
/// Sparse scatter helper for Adam8Bit (Dettmers et al., 2022 — 8-bit Adam
/// via block-wise quantization). The state schema is byte-packed
/// (signed-int8 for m, unsigned-uint8 for v) with a per-block FP64 scale
/// factor, so the sparse path can't just touch individual byte slots — it
/// has to operate at block granularity because changing a block's scale
/// re-interprets the byte codes of every element in that block.
/// </summary>
/// <remarks>
/// <para><b>Strategy.</b> Touched indices are grouped by block (block-id =
/// <c>idx / blockSize</c>). For each touched block:</para>
/// <list type="number">
/// <item>Dequantize the full block into a transient FP64 buffer.</item>
/// <item>Apply the Adam moment + parameter update at the touched indices
///     only — untouched mDeq / vDeq stay at their dequantized values.</item>
/// <item>Compute the block's new max-abs from the post-update mDeq / vDeq.</item>
/// <item>Re-quantize the full block with the new scales.</item>
/// </list>
/// <para>This makes the cost <c>O(blockSize · touchedBlocks)</c> rather than
/// <c>O(paramLen)</c>. For paper-default LayoutXLM (vocab=250 002, dim=768,
/// ~16 touched rows, blockSize=4096) that's ~12 blocks × 4 096 = 49 k
/// element-ops vs ~192 M dense — a ~4000× reduction.</para>
/// <para><b>Config eligibility.</b> The helper handles the most common
/// configuration: <c>CompressBothMoments=true</c>, <c>QuantizationPercentile≥100</c>
/// (absolute-max scale), no stochastic rounding. Falls back to the
/// ToDense path for the other configurations so quantization semantics
/// stay bit-identical.</para>
/// </remarks>
internal static partial class SparseEmbeddingOptimizerHelpers
{
    internal static bool TryApplyAdam8BitSparse<T>(
        Tensor<T> param,
        Vector<byte>? mQuantized, Vector<double>? mScales,
        Vector<byte> vQuantized, Vector<double> vScales,
        int blockSize, int numBlocks,
        double lr, double b1, double b2, double bc1, double bc2, double eps,
        bool compressBothMoments, double quantizationPercentile, bool useStochasticRounding)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (vQuantized is null) throw new ArgumentNullException(nameof(vQuantized));
        if (vScales is null) throw new ArgumentNullException(nameof(vScales));
        if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));

        // Eligibility — bail (caller falls through to dense ToDense path) on
        // configs we don't mirror bit-for-bit. CompressBothMoments=false isn't
        // a sparse blocker per se, but it means the m buffer is FP rather than
        // quantized — much less surface for the helper to win on, so defer it
        // to the FP path until profiling shows demand.
        if (!compressBothMoments) return false;
        if (mQuantized is null || mScales is null) return false;
        if (quantizationPercentile < 100.0) return false;
        if (useStochasticRounding) return false;

        var sparseList = DifferentiableOps.GetSparseEmbeddingGradsFor(param);
        if (sparseList is null || sparseList.Count == 0) return false;
        // Bail for multi-chunk: per-chunk block-dequant/requant would re-scan
        // the block's max-abs multiple times, changing the per-block scale
        // trajectory away from the dense path's single-step semantics.
        if (sparseList.Count != 1) return false;
        if (HasDuplicateRows(sparseList)) return false;
        if (param.Rank != 2) return false;
        int vocabSize = param.Shape[0];
        int embeddingDim = param.Shape[1];

        // Type fast paths — Adam8Bit's quantized state buffers are byte/double
        // regardless of T, so the only T-dependent piece is reading/writing
        // param values. Specialize the param span access for double and float;
        // generic T goes through NumOps.
        if (typeof(T) == typeof(double))
        {
            ApplyAdam8BitSparseDouble(param, mQuantized, mScales, vQuantized, vScales,
                sparseList, embeddingDim, vocabSize, blockSize, numBlocks,
                lr, b1, b2, bc1, bc2, eps);
            return true;
        }
        if (typeof(T) == typeof(float))
        {
            ApplyAdam8BitSparseFloat(param, mQuantized, mScales, vQuantized, vScales,
                sparseList, embeddingDim, vocabSize, blockSize, numBlocks,
                lr, b1, b2, bc1, bc2, eps);
            return true;
        }

        // Generic-T fallback.
        var ops = MathHelper.GetNumericOperations<T>();
        int paramLen = param.Length;

        // Identify touched blocks. Use a bit-set for fast lookup; numBlocks is
        // bounded by paramLen/blockSize so this is cheap.
        var touchedBlocks = new HashSet<int>();
        foreach (var sparse in sparseList)
        {
            int numIndices = sparse.NumIndices;
            for (int k = 0; k < numIndices; k++)
            {
                long row = sparse.Indices[k];
                if (row < 0 || row >= vocabSize) continue;
                int rowStart = (int)row * embeddingDim;
                int rowEnd = rowStart + embeddingDim;
                int firstBlock = rowStart / blockSize;
                int lastBlock = (rowEnd - 1) / blockSize;
                for (int b = firstBlock; b <= lastBlock; b++) touchedBlocks.Add(b);
            }
        }
        if (touchedBlocks.Count == 0) return true;

        // For each touched block: dequant → scatter Adam math → requant.
        // Use FP64 throughout — quantization scales are FP64 in the dense path.
        foreach (int b in touchedBlocks)
        {
            int blockStart = b * blockSize;
            int blockEnd = Math.Min(blockStart + blockSize, paramLen);
            int blockLen = blockEnd - blockStart;
            var mDeq = new double[blockLen];
            var vDeq = new double[blockLen];
            double mScale = mScales[b];
            double vScale = vScales[b];
            for (int i = 0; i < blockLen; i++)
            {
                mDeq[i] = (mQuantized[blockStart + i] - 128) * mScale;
                vDeq[i] = vQuantized[blockStart + i] * vScale;
            }

            // Apply Adam at touched indices within this block. We need to walk the
            // sparse list again to find indices that hit this specific block.
            foreach (var sparse in sparseList)
            {
                int numIndices = sparse.NumIndices;
                for (int k = 0; k < numIndices; k++)
                {
                    long row = sparse.Indices[k];
                    if (row < 0 || row >= vocabSize) continue;
                    int rowStart = (int)row * embeddingDim;
                    int valBase = k * embeddingDim;
                    for (int c = 0; c < embeddingDim; c++)
                    {
                        int flat = rowStart + c;
                        if (flat < blockStart || flat >= blockEnd) continue;
                        int local = flat - blockStart;
                        double g = ops.ToDouble(sparse.Values[valBase + c]);
                        double mNew = b1 * mDeq[local] + (1.0 - b1) * g;
                        double vNew = b2 * vDeq[local] + (1.0 - b2) * g * g;
                        mDeq[local] = mNew;
                        vDeq[local] = vNew;
                        double mHat = mNew / bc1;
                        double vHat = vNew / bc2;
                        double theta = ops.ToDouble(param[flat]);
                        theta -= lr * mHat / (Math.Sqrt(vHat) + eps);
                        param[flat] = ops.FromDouble(theta);
                    }
                }
            }

            // Recompute block scales from post-update max-abs.
            double mMax = 0.0, vMax = 0.0;
            for (int i = 0; i < blockLen; i++)
            {
                double a = Math.Abs(mDeq[i]); if (a > mMax) mMax = a;
                double s = Math.Abs(vDeq[i]); if (s > vMax) vMax = s;
            }
            double mScaleNew = mMax / 127.0; if (mScaleNew < 1e-10) mScaleNew = 1e-10;
            double vScaleNew = vMax / 255.0; if (vScaleNew < 1e-10) vScaleNew = 1e-10;
            mScales[b] = mScaleNew;
            vScales[b] = vScaleNew;

            // Re-quantize the whole block with new scales.
            for (int i = 0; i < blockLen; i++)
            {
                int qm = (int)Math.Round(mDeq[i] / mScaleNew);
                if (qm < -127) qm = -127; if (qm > 127) qm = 127;
                mQuantized[blockStart + i] = (byte)(qm + 128);

                int qv = (int)Math.Round(vDeq[i] / vScaleNew);
                if (qv < 0) qv = 0; if (qv > 255) qv = 255;
                vQuantized[blockStart + i] = (byte)qv;
            }
        }
        return true;
    }

    private static void ApplyAdam8BitSparseDouble<T>(
        Tensor<T> param,
        Vector<byte> mQuantized, Vector<double> mScales,
        Vector<byte> vQuantized, Vector<double> vScales,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList,
        int embeddingDim, int vocabSize, int blockSize, int numBlocks,
        double lr, double b1, double b2, double bc1, double bc2, double eps)
    {
        var paramSpan = ((Tensor<double>)(object)param).Data.Span;
        int paramLen = paramSpan.Length;
        double oneMinusB1 = 1.0 - b1;
        double oneMinusB2 = 1.0 - b2;

        var touchedBlocks = new HashSet<int>();
        foreach (var sparseObj in sparseList)
        {
            var sparse = (SparseEmbeddingGradient<double>)(object)sparseObj;
            int numIndices = sparse.NumIndices;
            for (int k = 0; k < numIndices; k++)
            {
                long row = sparse.Indices[k];
                if (row < 0 || row >= vocabSize) continue;
                int rowStart = (int)row * embeddingDim;
                int firstBlock = rowStart / blockSize;
                int lastBlock = (rowStart + embeddingDim - 1) / blockSize;
                for (int b = firstBlock; b <= lastBlock; b++) touchedBlocks.Add(b);
            }
        }
        if (touchedBlocks.Count == 0) return;

        foreach (int b in touchedBlocks)
        {
            int blockStart = b * blockSize;
            int blockEnd = Math.Min(blockStart + blockSize, paramLen);
            int blockLen = blockEnd - blockStart;
            var mDeq = new double[blockLen];
            var vDeq = new double[blockLen];
            double mScale = mScales[b];
            double vScale = vScales[b];
            for (int i = 0; i < blockLen; i++)
            {
                mDeq[i] = (mQuantized[blockStart + i] - 128) * mScale;
                vDeq[i] = vQuantized[blockStart + i] * vScale;
            }

            foreach (var sparseObj in sparseList)
            {
                var sparse = (SparseEmbeddingGradient<double>)(object)sparseObj;
                int numIndices = sparse.NumIndices;
                var valuesSpan = sparse.Values.Data.Span;
                for (int k = 0; k < numIndices; k++)
                {
                    long row = sparse.Indices[k];
                    if (row < 0 || row >= vocabSize) continue;
                    int rowStart = (int)row * embeddingDim;
                    int valBase = k * embeddingDim;
                    for (int c = 0; c < embeddingDim; c++)
                    {
                        int flat = rowStart + c;
                        if (flat < blockStart || flat >= blockEnd) continue;
                        int local = flat - blockStart;
                        double g = valuesSpan[valBase + c];
                        double mNew = b1 * mDeq[local] + oneMinusB1 * g;
                        double vNew = b2 * vDeq[local] + oneMinusB2 * g * g;
                        mDeq[local] = mNew;
                        vDeq[local] = vNew;
                        double mHat = mNew / bc1;
                        double vHat = vNew / bc2;
                        paramSpan[flat] -= lr * mHat / (Math.Sqrt(vHat) + eps);
                    }
                }
            }

            double mMax = 0.0, vMax = 0.0;
            for (int i = 0; i < blockLen; i++)
            {
                double a = Math.Abs(mDeq[i]); if (a > mMax) mMax = a;
                double s = Math.Abs(vDeq[i]); if (s > vMax) vMax = s;
            }
            double mScaleNew = mMax / 127.0; if (mScaleNew < 1e-10) mScaleNew = 1e-10;
            double vScaleNew = vMax / 255.0; if (vScaleNew < 1e-10) vScaleNew = 1e-10;
            mScales[b] = mScaleNew;
            vScales[b] = vScaleNew;

            for (int i = 0; i < blockLen; i++)
            {
                int qm = (int)Math.Round(mDeq[i] / mScaleNew);
                if (qm < -127) qm = -127; if (qm > 127) qm = 127;
                mQuantized[blockStart + i] = (byte)(qm + 128);

                int qv = (int)Math.Round(vDeq[i] / vScaleNew);
                if (qv < 0) qv = 0; if (qv > 255) qv = 255;
                vQuantized[blockStart + i] = (byte)qv;
            }
        }
    }

    private static void ApplyAdam8BitSparseFloat<T>(
        Tensor<T> param,
        Vector<byte> mQuantized, Vector<double> mScales,
        Vector<byte> vQuantized, Vector<double> vScales,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList,
        int embeddingDim, int vocabSize, int blockSize, int numBlocks,
        double lr, double b1, double b2, double bc1, double bc2, double eps)
    {
        var paramSpan = ((Tensor<float>)(object)param).Data.Span;
        int paramLen = paramSpan.Length;
        double oneMinusB1 = 1.0 - b1;
        double oneMinusB2 = 1.0 - b2;

        var touchedBlocks = new HashSet<int>();
        foreach (var sparseObj in sparseList)
        {
            var sparse = (SparseEmbeddingGradient<float>)(object)sparseObj;
            int numIndices = sparse.NumIndices;
            for (int k = 0; k < numIndices; k++)
            {
                long row = sparse.Indices[k];
                if (row < 0 || row >= vocabSize) continue;
                int rowStart = (int)row * embeddingDim;
                int firstBlock = rowStart / blockSize;
                int lastBlock = (rowStart + embeddingDim - 1) / blockSize;
                for (int b = firstBlock; b <= lastBlock; b++) touchedBlocks.Add(b);
            }
        }
        if (touchedBlocks.Count == 0) return;

        foreach (int b in touchedBlocks)
        {
            int blockStart = b * blockSize;
            int blockEnd = Math.Min(blockStart + blockSize, paramLen);
            int blockLen = blockEnd - blockStart;
            var mDeq = new double[blockLen];
            var vDeq = new double[blockLen];
            double mScale = mScales[b];
            double vScale = vScales[b];
            for (int i = 0; i < blockLen; i++)
            {
                mDeq[i] = (mQuantized[blockStart + i] - 128) * mScale;
                vDeq[i] = vQuantized[blockStart + i] * vScale;
            }

            foreach (var sparseObj in sparseList)
            {
                var sparse = (SparseEmbeddingGradient<float>)(object)sparseObj;
                int numIndices = sparse.NumIndices;
                var valuesSpan = sparse.Values.Data.Span;
                for (int k = 0; k < numIndices; k++)
                {
                    long row = sparse.Indices[k];
                    if (row < 0 || row >= vocabSize) continue;
                    int rowStart = (int)row * embeddingDim;
                    int valBase = k * embeddingDim;
                    for (int c = 0; c < embeddingDim; c++)
                    {
                        int flat = rowStart + c;
                        if (flat < blockStart || flat >= blockEnd) continue;
                        int local = flat - blockStart;
                        double g = valuesSpan[valBase + c];
                        double mNew = b1 * mDeq[local] + oneMinusB1 * g;
                        double vNew = b2 * vDeq[local] + oneMinusB2 * g * g;
                        mDeq[local] = mNew;
                        vDeq[local] = vNew;
                        double mHat = mNew / bc1;
                        double vHat = vNew / bc2;
                        paramSpan[flat] = (float)(paramSpan[flat] - lr * mHat / (Math.Sqrt(vHat) + eps));
                    }
                }
            }

            double mMax = 0.0, vMax = 0.0;
            for (int i = 0; i < blockLen; i++)
            {
                double a = Math.Abs(mDeq[i]); if (a > mMax) mMax = a;
                double s = Math.Abs(vDeq[i]); if (s > vMax) vMax = s;
            }
            double mScaleNew = mMax / 127.0; if (mScaleNew < 1e-10) mScaleNew = 1e-10;
            double vScaleNew = vMax / 255.0; if (vScaleNew < 1e-10) vScaleNew = 1e-10;
            mScales[b] = mScaleNew;
            vScales[b] = vScaleNew;

            for (int i = 0; i < blockLen; i++)
            {
                int qm = (int)Math.Round(mDeq[i] / mScaleNew);
                if (qm < -127) qm = -127; if (qm > 127) qm = 127;
                mQuantized[blockStart + i] = (byte)(qm + 128);

                int qv = (int)Math.Round(vDeq[i] / vScaleNew);
                if (qv < 0) qv = 0; if (qv > 255) qv = 255;
                vQuantized[blockStart + i] = (byte)qv;
            }
        }
    }
}
