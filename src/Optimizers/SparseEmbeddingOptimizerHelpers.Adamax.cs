using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Optimizers;

/// <summary>
/// Sparse scatter helper for Adamax — Adam with the L∞ norm (infinity-norm /
/// max) in place of the L2-based v. The per-element state is
/// <c>u ← max(β2·u, |g|)</c>; the step is <c>θ ← θ - (lr/bc1)·m / (u + ε)</c>.
/// </summary>
internal static partial class SparseEmbeddingOptimizerHelpers
{
    internal static bool TryApplyAdamaxSparse<T>(
        Tensor<T> param, Tensor<T> m, Tensor<T> u,
        double lr, double b1, double b2, double bc1, double eps, double weightDecay)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (u is null) throw new ArgumentNullException(nameof(u));

        var sparseList = DifferentiableOps.GetSparseEmbeddingGradsFor(param);
        if (sparseList is null || sparseList.Count == 0) return false;
        // Bail: multi-chunk advances Adamax state once per chunk instead of on
        // the summed gradient; non-zero weight decay would skip untouched rows
        // in dense Adamax.
        if (sparseList.Count != 1) return false;
        if (weightDecay != 0.0) return false;
        if (HasDuplicateRows(sparseList)) return false;
        if (param.Rank != 2 || m.Rank != 2 || u.Rank != 2) return false;
        int vocabSize = param.Shape[0];
        int embeddingDim = param.Shape[1];
        if (m.Shape[0] != vocabSize || m.Shape[1] != embeddingDim) return false;
        if (u.Shape[0] != vocabSize || u.Shape[1] != embeddingDim) return false;

        double oneMinusB1 = 1.0 - b1;
        double lrAdj = lr / bc1;

        if (typeof(T) == typeof(double))
        {
            ApplyAdamaxSparseDouble(param, m, u, sparseList, embeddingDim,
                lrAdj, b1, oneMinusB1, b2, eps, weightDecay);
            return true;
        }
        if (typeof(T) == typeof(float))
        {
            ApplyAdamaxSparseFloat(param, m, u, sparseList, embeddingDim,
                (float)lrAdj, (float)b1, (float)oneMinusB1, (float)b2, (float)eps, (float)weightDecay);
            return true;
        }

        var ops = MathHelper.GetNumericOperations<T>();
        T b1T = ops.FromDouble(b1), omB1 = ops.FromDouble(oneMinusB1);
        T b2T = ops.FromDouble(b2);
        T lrAdjT = ops.FromDouble(lrAdj);
        T epsT = ops.FromDouble(eps), wdT = ops.FromDouble(weightDecay);
        bool hasWd = weightDecay > 0.0;

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
                    T g = values[valBase + c];
                    T theta = param[paramBase + c];
                    if (hasWd) g = ops.Add(g, ops.Multiply(wdT, theta));
                    T mNew = ops.Add(ops.Multiply(b1T, m[paramBase + c]), ops.Multiply(omB1, g));
                    T uOld = u[paramBase + c];
                    T absG = ops.Abs(g);
                    T b2u = ops.Multiply(b2T, uOld);
                    T uNew = ops.GreaterThan(b2u, absG) ? b2u : absG;
                    m[paramBase + c] = mNew;
                    u[paramBase + c] = uNew;
                    param[paramBase + c] = ops.Subtract(theta,
                        ops.Divide(ops.Multiply(lrAdjT, mNew), ops.Add(uNew, epsT)));
                }
            }
        }
        return true;
    }

    private static void ApplyAdamaxSparseDouble<T>(
        Tensor<T> param, Tensor<T> m, Tensor<T> u,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        double lrAdj, double b1, double oneMinusB1, double b2, double eps, double weightDecay)
    {
        var paramSpan = ((Tensor<double>)(object)param).Data.Span;
        var mSpan = ((Tensor<double>)(object)m).Data.Span;
        var uSpan = ((Tensor<double>)(object)u).Data.Span;
        bool hasWd = weightDecay > 0.0;
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
                    double theta = paramSpan[paramBase + c];
                    if (hasWd) g += weightDecay * theta;
                    double mNew = b1 * mSpan[paramBase + c] + oneMinusB1 * g;
                    double uNew = Math.Max(b2 * uSpan[paramBase + c], Math.Abs(g));
                    mSpan[paramBase + c] = mNew;
                    uSpan[paramBase + c] = uNew;
                    paramSpan[paramBase + c] = theta - lrAdj * mNew / (uNew + eps);
                }
            }
        }
    }

    private static void ApplyAdamaxSparseFloat<T>(
        Tensor<T> param, Tensor<T> m, Tensor<T> u,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        float lrAdj, float b1, float oneMinusB1, float b2, float eps, float weightDecay)
    {
        var paramSpan = ((Tensor<float>)(object)param).Data.Span;
        var mSpan = ((Tensor<float>)(object)m).Data.Span;
        var uSpan = ((Tensor<float>)(object)u).Data.Span;
        bool hasWd = weightDecay > 0f;
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
                    float theta = paramSpan[paramBase + c];
                    if (hasWd) g += weightDecay * theta;
                    float mNew = b1 * mSpan[paramBase + c] + oneMinusB1 * g;
                    float uNew = Math.Max(b2 * uSpan[paramBase + c], Math.Abs(g));
                    mSpan[paramBase + c] = mNew;
                    uSpan[paramBase + c] = uNew;
                    paramSpan[paramBase + c] = theta - lrAdj * mNew / (uNew + eps);
                }
            }
        }
    }
}
