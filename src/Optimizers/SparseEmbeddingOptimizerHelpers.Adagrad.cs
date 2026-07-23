using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Optimizers;

/// <summary>
/// Sparse scatter helper for Adagrad. Maintains a per-element accumulator of
/// squared gradients; the parameter step at each index is
/// <c>θ ← θ - lr·g / (√accum + ε)</c>. Sparse path updates accum + θ at touched
/// embedding-table rows only.
/// </summary>
internal static partial class SparseEmbeddingOptimizerHelpers
{
    internal static bool TryApplyAdagradSparse<T>(
        Tensor<T> param, Tensor<T> accum,
        double lr, double eps, double weightDecay)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (accum is null) throw new ArgumentNullException(nameof(accum));

        var sparseList = DifferentiableOps.GetSparseEmbeddingGradsFor(param);
        if (sparseList is null || sparseList.Count == 0) return false;
        // Bail: accum += g² is additive but per-chunk advances the accumulator
        // separately for each chunk; on a same-row repeat the accumulator gets
        // bumped twice with each individual g² instead of once with (g₁+g₂)².
        // Non-zero weight decay would skip untouched rows.
        if (sparseList.Count != 1) return false;
        if (weightDecay != 0.0) return false;
        if (HasDuplicateRows(sparseList)) return false;
        if (param.Rank != 2 || accum.Rank != 2) return false;
        int vocabSize = param.Shape[0];
        int embeddingDim = param.Shape[1];
        if (accum.Shape[0] != vocabSize || accum.Shape[1] != embeddingDim) return false;

        if (typeof(T) == typeof(double))
        {
            ApplyAdagradSparseDouble(param, accum, sparseList, embeddingDim, lr, eps, weightDecay);
            return true;
        }
        if (typeof(T) == typeof(float))
        {
            ApplyAdagradSparseFloat(param, accum, sparseList, embeddingDim,
                (float)lr, (float)eps, (float)weightDecay);
            return true;
        }

        var ops = MathHelper.GetNumericOperations<T>();
        T lrT = ops.FromDouble(lr), epsT = ops.FromDouble(eps), wdT = ops.FromDouble(weightDecay);
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
                    T aNew = ops.Add(accum[paramBase + c], ops.Multiply(g, g));
                    accum[paramBase + c] = aNew;
                    T denom = ops.Add(ops.Sqrt(aNew), epsT);
                    param[paramBase + c] = ops.Subtract(theta, ops.Divide(ops.Multiply(lrT, g), denom));
                }
            }
        }
        return true;
    }

    private static void ApplyAdagradSparseDouble<T>(
        Tensor<T> param, Tensor<T> accum,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        double lr, double eps, double weightDecay)
    {
        var paramSpan = ((Tensor<double>)(object)param).Data.Span;
        var accSpan = ((Tensor<double>)(object)accum).Data.Span;
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
                    double aNew = accSpan[paramBase + c] + g * g;
                    accSpan[paramBase + c] = aNew;
                    paramSpan[paramBase + c] = theta - lr * g / (Math.Sqrt(aNew) + eps);
                }
            }
        }
    }

    private static void ApplyAdagradSparseFloat<T>(
        Tensor<T> param, Tensor<T> accum,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        float lr, float eps, float weightDecay)
    {
        var paramSpan = ((Tensor<float>)(object)param).Data.Span;
        var accSpan = ((Tensor<float>)(object)accum).Data.Span;
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
                    float aNew = accSpan[paramBase + c] + g * g;
                    accSpan[paramBase + c] = aNew;
                    paramSpan[paramBase + c] = theta - lr * g / ((float)Math.Sqrt(aNew) + eps);
                }
            }
        }
    }
}
