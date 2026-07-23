using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Optimizers;

/// <summary>
/// Sparse scatter helper for AdaDelta (Zeiler, 2012). Maintains EMA of
/// squared gradients (accumGrad) and EMA of squared updates (accumDelta);
/// each step computes <c>dx = sqrt(accumDelta + ε) / sqrt(accumGrad + ε) · g</c>
/// and updates both accumulators.
/// </summary>
internal static partial class SparseEmbeddingOptimizerHelpers
{
    internal static bool TryApplyAdaDeltaSparse<T>(
        Tensor<T> param, Tensor<T> accumGrad, Tensor<T> accumDelta,
        double lr, double rho, double eps, double weightDecay)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (accumGrad is null) throw new ArgumentNullException(nameof(accumGrad));
        if (accumDelta is null) throw new ArgumentNullException(nameof(accumDelta));

        var sparseList = DifferentiableOps.GetSparseEmbeddingGradsFor(param);
        if (sparseList is null || sparseList.Count == 0) return false;
        // Bail: AdaDelta's adaptive EMAs assume one update per step over the
        // summed gradient. Per-chunk processing would update both accumulators
        // (accumGrad, accumDelta) multiple times for a single logical step.
        // Non-zero weight decay would also skip untouched rows.
        if (sparseList.Count != 1) return false;
        if (weightDecay != 0.0) return false;
        if (HasDuplicateRows(sparseList)) return false;
        if (param.Rank != 2 || accumGrad.Rank != 2 || accumDelta.Rank != 2) return false;
        int vocabSize = param.Shape[0];
        int embeddingDim = param.Shape[1];
        if (accumGrad.Shape[0] != vocabSize || accumGrad.Shape[1] != embeddingDim) return false;
        if (accumDelta.Shape[0] != vocabSize || accumDelta.Shape[1] != embeddingDim) return false;

        double oneMinusRho = 1.0 - rho;

        if (typeof(T) == typeof(double))
        {
            ApplyAdaDeltaSparseDouble(param, accumGrad, accumDelta, sparseList, embeddingDim,
                lr, rho, oneMinusRho, eps, weightDecay);
            return true;
        }
        if (typeof(T) == typeof(float))
        {
            ApplyAdaDeltaSparseFloat(param, accumGrad, accumDelta, sparseList, embeddingDim,
                (float)lr, (float)rho, (float)oneMinusRho, (float)eps, (float)weightDecay);
            return true;
        }

        var ops = MathHelper.GetNumericOperations<T>();
        T rhoT = ops.FromDouble(rho), omRho = ops.FromDouble(oneMinusRho);
        T epsT = ops.FromDouble(eps), lrT = ops.FromDouble(lr), wdT = ops.FromDouble(weightDecay);
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
                    T agOld = accumGrad[paramBase + c];
                    T agNew = ops.Add(ops.Multiply(rhoT, agOld), ops.Multiply(omRho, ops.Multiply(g, g)));
                    accumGrad[paramBase + c] = agNew;
                    T adOld = accumDelta[paramBase + c];
                    T dx = ops.Multiply(
                        ops.Divide(ops.Sqrt(ops.Add(adOld, epsT)), ops.Sqrt(ops.Add(agNew, epsT))),
                        g);
                    accumDelta[paramBase + c] = ops.Add(ops.Multiply(rhoT, adOld),
                                                        ops.Multiply(omRho, ops.Multiply(dx, dx)));
                    param[paramBase + c] = ops.Subtract(theta, ops.Multiply(lrT, dx));
                }
            }
        }
        return true;
    }

    private static void ApplyAdaDeltaSparseDouble<T>(
        Tensor<T> param, Tensor<T> accumGrad, Tensor<T> accumDelta,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        double lr, double rho, double oneMinusRho, double eps, double weightDecay)
    {
        var paramSpan = ((Tensor<double>)(object)param).Data.Span;
        var agSpan = ((Tensor<double>)(object)accumGrad).Data.Span;
        var adSpan = ((Tensor<double>)(object)accumDelta).Data.Span;
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
                    double agNew = rho * agSpan[paramBase + c] + oneMinusRho * g * g;
                    agSpan[paramBase + c] = agNew;
                    double dx = Math.Sqrt(adSpan[paramBase + c] + eps) / Math.Sqrt(agNew + eps) * g;
                    adSpan[paramBase + c] = rho * adSpan[paramBase + c] + oneMinusRho * dx * dx;
                    paramSpan[paramBase + c] = theta - lr * dx;
                }
            }
        }
    }

    private static void ApplyAdaDeltaSparseFloat<T>(
        Tensor<T> param, Tensor<T> accumGrad, Tensor<T> accumDelta,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        float lr, float rho, float oneMinusRho, float eps, float weightDecay)
    {
        var paramSpan = ((Tensor<float>)(object)param).Data.Span;
        var agSpan = ((Tensor<float>)(object)accumGrad).Data.Span;
        var adSpan = ((Tensor<float>)(object)accumDelta).Data.Span;
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
                    float agNew = rho * agSpan[paramBase + c] + oneMinusRho * g * g;
                    agSpan[paramBase + c] = agNew;
                    float dx = (float)Math.Sqrt(adSpan[paramBase + c] + eps) /
                               (float)Math.Sqrt(agNew + eps) * g;
                    adSpan[paramBase + c] = rho * adSpan[paramBase + c] + oneMinusRho * dx * dx;
                    paramSpan[paramBase + c] = theta - lr * dx;
                }
            }
        }
    }
}
