using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Optimizers;

/// <summary>
/// Sparse scatter helper for RMSProp (Hinton lecture). Maintains an EMA of
/// squared gradients per element; sparse path updates the EMA and parameter
/// at touched embedding-table rows only.
/// </summary>
internal static partial class SparseEmbeddingOptimizerHelpers
{
    internal static bool TryApplyRmspropSparse<T>(
        Tensor<T> param, Tensor<T> squaredAvg,
        double lr, double rho, double eps, double weightDecay)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (squaredAvg is null) throw new ArgumentNullException(nameof(squaredAvg));

        var sparseList = DifferentiableOps.GetSparseEmbeddingGradsFor(param);
        if (sparseList is null || sparseList.Count == 0) return false;
        // Bail: squaredAvg EMA assumes one update per logical step on the summed
        // gradient; per-chunk processing would advance the EMA multiple times.
        // Non-zero weight decay would skip untouched rows.
        if (sparseList.Count != 1) return false;
        if (weightDecay != 0.0) return false;
        if (HasDuplicateRows(sparseList)) return false;
        if (param.Rank != 2 || squaredAvg.Rank != 2) return false;
        int vocabSize = param.Shape[0];
        int embeddingDim = param.Shape[1];
        if (squaredAvg.Shape[0] != vocabSize || squaredAvg.Shape[1] != embeddingDim) return false;

        double oneMinusRho = 1.0 - rho;

        if (typeof(T) == typeof(double))
        {
            ApplyRmspropSparseDouble(param, squaredAvg, sparseList, embeddingDim,
                lr, rho, oneMinusRho, eps, weightDecay);
            return true;
        }
        if (typeof(T) == typeof(float))
        {
            ApplyRmspropSparseFloat(param, squaredAvg, sparseList, embeddingDim,
                (float)lr, (float)rho, (float)oneMinusRho, (float)eps, (float)weightDecay);
            return true;
        }

        var ops = MathHelper.GetNumericOperations<T>();
        T lrT = ops.FromDouble(lr), rhoT = ops.FromDouble(rho), omRho = ops.FromDouble(oneMinusRho);
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
                    T sqOld = squaredAvg[paramBase + c];
                    T sqNew = ops.Add(ops.Multiply(rhoT, sqOld), ops.Multiply(omRho, ops.Multiply(g, g)));
                    squaredAvg[paramBase + c] = sqNew;
                    T denom = ops.Add(ops.Sqrt(sqNew), epsT);
                    param[paramBase + c] = ops.Subtract(theta, ops.Divide(ops.Multiply(lrT, g), denom));
                }
            }
        }
        return true;
    }

    private static void ApplyRmspropSparseDouble<T>(
        Tensor<T> param, Tensor<T> squaredAvg,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        double lr, double rho, double oneMinusRho, double eps, double weightDecay)
    {
        var paramSpan = ((Tensor<double>)(object)param).Data.Span;
        var sqSpan = ((Tensor<double>)(object)squaredAvg).Data.Span;
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
                    double sqNew = rho * sqSpan[paramBase + c] + oneMinusRho * g * g;
                    sqSpan[paramBase + c] = sqNew;
                    paramSpan[paramBase + c] = theta - lr * g / (Math.Sqrt(sqNew) + eps);
                }
            }
        }
    }

    private static void ApplyRmspropSparseFloat<T>(
        Tensor<T> param, Tensor<T> squaredAvg,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        float lr, float rho, float oneMinusRho, float eps, float weightDecay)
    {
        var paramSpan = ((Tensor<float>)(object)param).Data.Span;
        var sqSpan = ((Tensor<float>)(object)squaredAvg).Data.Span;
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
                    float sqNew = rho * sqSpan[paramBase + c] + oneMinusRho * g * g;
                    sqSpan[paramBase + c] = sqNew;
                    paramSpan[paramBase + c] = theta - lr * g / ((float)Math.Sqrt(sqNew) + eps);
                }
            }
        }
    }
}
