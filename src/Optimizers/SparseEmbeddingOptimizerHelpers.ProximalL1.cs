using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Optimizers;

/// <summary>
/// Sparse scatter helper for proximal-gradient (ISTA) with L1 soft-threshold.
/// Update: <c>p_after = θ - lr·g; θ' = sign(p_after)·max(|p_after| - lr·l1, 0)</c>.
/// </summary>
internal static partial class SparseEmbeddingOptimizerHelpers
{
    internal static bool TryApplyProximalL1Sparse<T>(
        Tensor<T> param,
        double lr, double l1Strength)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));

        var sparseList = DifferentiableOps.GetSparseEmbeddingGradsFor(param);
        if (sparseList is null || sparseList.Count == 0) return false;
        // Bail for multi-chunk: dense proximal-L1 first sums all gradient
        // contributions then applies the proximal operator once; per-chunk
        // would shift the threshold boundary on the same row multiple times.
        if (sparseList.Count != 1) return false;
        if (HasDuplicateRows(sparseList)) return false;
        if (param.Rank != 2) return false;
        int vocabSize = param.Shape[0];
        int embeddingDim = param.Shape[1];

        double threshold = lr * l1Strength;

        if (typeof(T) == typeof(double))
        {
            ApplyProximalL1SparseDouble(param, sparseList, embeddingDim, lr, threshold);
            return true;
        }
        if (typeof(T) == typeof(float))
        {
            ApplyProximalL1SparseFloat(param, sparseList, embeddingDim, (float)lr, (float)threshold);
            return true;
        }

        var ops = MathHelper.GetNumericOperations<T>();
        T lrT = ops.FromDouble(lr);
        T threshT = ops.FromDouble(threshold);
        T negThreshT = ops.Negate(threshT);
        T zero = ops.Zero;

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
                    T pAfter = ops.Subtract(param[paramBase + c], ops.Multiply(lrT, g));
                    T result;
                    if (ops.GreaterThan(pAfter, threshT)) result = ops.Subtract(pAfter, threshT);
                    else if (ops.LessThan(pAfter, negThreshT)) result = ops.Add(pAfter, threshT);
                    else result = zero;
                    param[paramBase + c] = result;
                }
            }
        }
        return true;
    }

    private static void ApplyProximalL1SparseDouble<T>(
        Tensor<T> param,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        double lr, double threshold)
    {
        var paramSpan = ((Tensor<double>)(object)param).Data.Span;
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
                    double pAfter = paramSpan[paramBase + c] - lr * g;
                    if (pAfter > threshold) paramSpan[paramBase + c] = pAfter - threshold;
                    else if (pAfter < -threshold) paramSpan[paramBase + c] = pAfter + threshold;
                    else paramSpan[paramBase + c] = 0.0;
                }
            }
        }
    }

    private static void ApplyProximalL1SparseFloat<T>(
        Tensor<T> param,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        float lr, float threshold)
    {
        var paramSpan = ((Tensor<float>)(object)param).Data.Span;
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
                    float pAfter = paramSpan[paramBase + c] - lr * g;
                    if (pAfter > threshold) paramSpan[paramBase + c] = pAfter - threshold;
                    else if (pAfter < -threshold) paramSpan[paramBase + c] = pAfter + threshold;
                    else paramSpan[paramBase + c] = 0f;
                }
            }
        }
    }
}
