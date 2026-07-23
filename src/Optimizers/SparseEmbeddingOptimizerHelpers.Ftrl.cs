using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Optimizers;

/// <summary>
/// Sparse scatter helper for FTRL (McMahan, 2013 — Follow-The-Regularized-Leader).
/// Per-element accumulators z and n; exact L1 soft-thresholding produces sparse
/// weights. Sigma update assumes <c>lrPower &lt; 0</c> (typical <c>-0.5</c> for
/// the sqrt(n) schedule).
/// </summary>
internal static partial class SparseEmbeddingOptimizerHelpers
{
    internal static bool TryApplyFtrlSparse<T>(
        Tensor<T> param, Tensor<T> z, Tensor<T> n,
        double lr, double l1Reg, double l2Reg, double lrPower)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (z is null) throw new ArgumentNullException(nameof(z));
        if (n is null) throw new ArgumentNullException(nameof(n));

        var sparseList = DifferentiableOps.GetSparseEmbeddingGradsFor(param);
        if (sparseList is null || sparseList.Count == 0) return false;
        // Bail for multi-chunk: FTRL's z + n accumulators update per chunk
        // instead of per logical step.
        if (sparseList.Count != 1) return false;
        if (HasDuplicateRows(sparseList)) return false;
        if (param.Rank != 2 || z.Rank != 2 || n.Rank != 2) return false;
        int vocabSize = param.Shape[0];
        int embeddingDim = param.Shape[1];
        if (z.Shape[0] != vocabSize || z.Shape[1] != embeddingDim) return false;
        if (n.Shape[0] != vocabSize || n.Shape[1] != embeddingDim) return false;

        if (typeof(T) == typeof(double))
        {
            ApplyFtrlSparseDouble(param, z, n, sparseList, embeddingDim, lr, l1Reg, l2Reg, lrPower);
            return true;
        }
        if (typeof(T) == typeof(float))
        {
            ApplyFtrlSparseFloat(param, z, n, sparseList, embeddingDim,
                (float)lr, (float)l1Reg, (float)l2Reg, (float)lrPower);
            return true;
        }

        // Generic-T fallback: the FTRL math has Pow(n, -lrPower) which doesn't
        // map cleanly through INumericOperations, so go through double internally.
        var ops = MathHelper.GetNumericOperations<T>();
        T zeroT = ops.Zero;
        foreach (var sparse in sparseList)
        {
            int numIndices = sparse.NumIndices;
            if (numIndices == 0) continue;
            var values = sparse.Values;
            var indices = sparse.Indices;
            for (int k = 0; k < numIndices; k++)
            {
                long row = indices[k];
                if (row < 0 || row >= vocabSize) continue;
                int paramBase = (int)row * embeddingDim;
                int valBase = k * embeddingDim;
                for (int c = 0; c < embeddingDim; c++)
                {
                    double g = ops.ToDouble(values[valBase + c]);
                    double nOld = ops.ToDouble(n[paramBase + c]);
                    double nNew = nOld + g * g;
                    double sigma = (Math.Pow(nNew, -lrPower) - Math.Pow(nOld, -lrPower)) / lr;
                    if (double.IsNaN(sigma) || double.IsInfinity(sigma)) sigma = 0.0;
                    double zNew = ops.ToDouble(z[paramBase + c]) + g - sigma * ops.ToDouble(param[paramBase + c]);
                    n[paramBase + c] = ops.FromDouble(nNew);
                    z[paramBase + c] = ops.FromDouble(zNew);
                    if (Math.Abs(zNew) <= l1Reg) { param[paramBase + c] = zeroT; continue; }
                    double pre = Math.Pow(nNew, -lrPower) / lr + l2Reg;
                    param[paramBase + c] = ops.FromDouble((Math.Sign(zNew) * l1Reg - zNew) / pre);
                }
            }
        }
        return true;
    }

    private static void ApplyFtrlSparseDouble<T>(
        Tensor<T> param, Tensor<T> z, Tensor<T> n,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        double lr, double l1Reg, double l2Reg, double lrPower)
    {
        var paramSpan = ((Tensor<double>)(object)param).Data.Span;
        var zSpan = ((Tensor<double>)(object)z).Data.Span;
        var nSpan = ((Tensor<double>)(object)n).Data.Span;
        int vocabSize = paramSpan.Length / embeddingDim;

        foreach (var sparseObj in sparseList)
        {
            var sparse = (SparseEmbeddingGradient<double>)(object)sparseObj;
            int numIndices = sparse.NumIndices;
            if (numIndices == 0) continue;
            var valuesSpan = sparse.Values.Data.Span;
            for (int k = 0; k < numIndices; k++)
            {
                long row = sparse.Indices[k];
                if (row < 0 || row >= vocabSize) continue;
                int paramBase = (int)row * embeddingDim;
                int valBase = k * embeddingDim;
                for (int c = 0; c < embeddingDim; c++)
                {
                    double g = valuesSpan[valBase + c];
                    double nOld = nSpan[paramBase + c];
                    double nNew = nOld + g * g;
                    double sigma = (Math.Pow(nNew, -lrPower) - Math.Pow(nOld, -lrPower)) / lr;
                    if (double.IsNaN(sigma) || double.IsInfinity(sigma)) sigma = 0.0;
                    double zNew = zSpan[paramBase + c] + g - sigma * paramSpan[paramBase + c];
                    nSpan[paramBase + c] = nNew;
                    zSpan[paramBase + c] = zNew;
                    if (Math.Abs(zNew) <= l1Reg) { paramSpan[paramBase + c] = 0.0; continue; }
                    double pre = Math.Pow(nNew, -lrPower) / lr + l2Reg;
                    paramSpan[paramBase + c] = (Math.Sign(zNew) * l1Reg - zNew) / pre;
                }
            }
        }
    }

    private static void ApplyFtrlSparseFloat<T>(
        Tensor<T> param, Tensor<T> z, Tensor<T> n,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        float lr, float l1Reg, float l2Reg, float lrPower)
    {
        var paramSpan = ((Tensor<float>)(object)param).Data.Span;
        var zSpan = ((Tensor<float>)(object)z).Data.Span;
        var nSpan = ((Tensor<float>)(object)n).Data.Span;
        int vocabSize = paramSpan.Length / embeddingDim;

        foreach (var sparseObj in sparseList)
        {
            var sparse = (SparseEmbeddingGradient<float>)(object)sparseObj;
            int numIndices = sparse.NumIndices;
            if (numIndices == 0) continue;
            var valuesSpan = sparse.Values.Data.Span;
            for (int k = 0; k < numIndices; k++)
            {
                long row = sparse.Indices[k];
                if (row < 0 || row >= vocabSize) continue;
                int paramBase = (int)row * embeddingDim;
                int valBase = k * embeddingDim;
                for (int c = 0; c < embeddingDim; c++)
                {
                    float g = valuesSpan[valBase + c];
                    float nOld = nSpan[paramBase + c];
                    float nNew = nOld + g * g;
                    float sigma = ((float)Math.Pow(nNew, -lrPower) - (float)Math.Pow(nOld, -lrPower)) / lr;
                    if (float.IsNaN(sigma) || float.IsInfinity(sigma)) sigma = 0f;
                    float zNew = zSpan[paramBase + c] + g - sigma * paramSpan[paramBase + c];
                    nSpan[paramBase + c] = nNew;
                    zSpan[paramBase + c] = zNew;
                    if (Math.Abs(zNew) <= l1Reg) { paramSpan[paramBase + c] = 0f; continue; }
                    float pre = (float)Math.Pow(nNew, -lrPower) / lr + l2Reg;
                    paramSpan[paramBase + c] = (Math.Sign(zNew) * l1Reg - zNew) / pre;
                }
            }
        }
    }
}
