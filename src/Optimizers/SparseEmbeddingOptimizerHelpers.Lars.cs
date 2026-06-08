using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Optimizers;

/// <summary>
/// Sparse scatter helper for LARS (You et al., 2017 — Layer-wise Adaptive
/// Rate Scaling). Computes <c>‖p‖₂</c> over the full parameter (one O(N)
/// reduction); <c>‖g‖₂²</c> is the sum of squared values across touched
/// indices only — exact equality with the dense ‖g‖₂² since untouched
/// gradients are zero. Velocity + parameter updates scatter at touched
/// indices only.
/// </summary>
internal static partial class SparseEmbeddingOptimizerHelpers
{
    internal static bool TryApplyLarsSparse<T>(
        Tensor<T> param, Tensor<T> velocity,
        double lr, double momentum, double weightDecay, double trustCoeff, double eps)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (velocity is null) throw new ArgumentNullException(nameof(velocity));

        var sparseList = DifferentiableOps.GetSparseEmbeddingGradsFor(param);
        if (sparseList is null || sparseList.Count == 0) return false;
        // Bail to dense for multi-chunk grads (gNormSq otherwise accumulates per
        // chunk instead of over the summed gradient) and for non-zero weight
        // decay (dense LARS decays every parameter index; sparse here would
        // only decay touched rows, so untouched embedding rows stop decaying).
        if (sparseList.Count != 1) return false;
        if (weightDecay != 0.0) return false;
        if (HasDuplicateRows(sparseList)) return false;
        if (param.Rank != 2 || velocity.Rank != 2) return false;
        int vocabSize = param.Shape[0];
        int embeddingDim = param.Shape[1];
        if (velocity.Shape[0] != vocabSize || velocity.Shape[1] != embeddingDim) return false;

        if (typeof(T) == typeof(double))
        {
            ApplyLarsSparseDouble(param, velocity, sparseList, embeddingDim,
                lr, momentum, weightDecay, trustCoeff, eps);
            return true;
        }
        if (typeof(T) == typeof(float))
        {
            ApplyLarsSparseFloat(param, velocity, sparseList, embeddingDim,
                (float)lr, (float)momentum, (float)weightDecay, (float)trustCoeff, (float)eps);
            return true;
        }

        // Generic-T fallback.
        var ops = MathHelper.GetNumericOperations<T>();
        // ‖p‖₂.
        double pNormSq = 0.0;
        for (int i = 0; i < param.Length; i++) { double pi = ops.ToDouble(param[i]); pNormSq += pi * pi; }
        double pNorm = Math.Sqrt(pNormSq);
        // ‖g‖₂² over touched (= dense ‖g‖₂² since untouched g = 0).
        double gNormSq = 0.0;
        foreach (var s in sparseList)
        {
            int n = s.NumIndices * embeddingDim;
            for (int k = 0; k < n; k++) { double gd = ops.ToDouble(s.Values[k]); gNormSq += gd * gd; }
        }
        double gNorm = Math.Sqrt(gNormSq);
        // Match the dense LARS small-norm guard: when either norm is below eps,
        // fall back to the base learning rate to avoid a degenerate trust ratio.
        // Otherwise lars-scale by ‖p‖ / (‖g‖ + wd·‖p‖ + eps).
        double localLr;
        if (pNorm < eps || gNorm < eps) localLr = lr;
        else localLr = lr * trustCoeff * pNorm / (gNorm + weightDecay * pNorm + eps);
        T localLrT = ops.FromDouble(localLr);
        T momT = ops.FromDouble(momentum);
        T wdT = ops.FromDouble(weightDecay);
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
                    T vOld = velocity[paramBase + c];
                    T vNew = ops.Add(ops.Multiply(momT, vOld), ops.Multiply(localLrT, g));
                    velocity[paramBase + c] = vNew;
                    param[paramBase + c] = ops.Subtract(theta, vNew);
                }
            }
        }
        return true;
    }

    private static void ApplyLarsSparseDouble<T>(
        Tensor<T> param, Tensor<T> velocity,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        double lr, double momentum, double weightDecay, double trustCoeff, double eps)
    {
        var paramSpan = ((Tensor<double>)(object)param).Data.Span;
        var velSpan = ((Tensor<double>)(object)velocity).Data.Span;
        bool hasWd = weightDecay > 0.0;
        int vocabSize = paramSpan.Length / embeddingDim;

        double pNormSq = 0.0;
        for (int i = 0; i < paramSpan.Length; i++) pNormSq += paramSpan[i] * paramSpan[i];
        double pNorm = Math.Sqrt(pNormSq);
        double gNormSq = 0.0;
        foreach (var sparseObj in sparseList)
        {
            var sparse = (SparseEmbeddingGradient<double>)(object)sparseObj;
            int n = sparse.NumIndices * embeddingDim;
            var vs = sparse.Values.Data.Span;
            for (int k = 0; k < n; k++) gNormSq += vs[k] * vs[k];
        }
        double gNorm = Math.Sqrt(gNormSq);
        double localLr;
        if (pNorm < eps || gNorm < eps) localLr = lr;
        else localLr = lr * trustCoeff * pNorm / (gNorm + weightDecay * pNorm + eps);

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
                    double vNew = momentum * velSpan[paramBase + c] + localLr * g;
                    velSpan[paramBase + c] = vNew;
                    paramSpan[paramBase + c] = theta - vNew;
                }
            }
        }
    }

    private static void ApplyLarsSparseFloat<T>(
        Tensor<T> param, Tensor<T> velocity,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        float lr, float momentum, float weightDecay, float trustCoeff, float eps)
    {
        var paramSpan = ((Tensor<float>)(object)param).Data.Span;
        var velSpan = ((Tensor<float>)(object)velocity).Data.Span;
        bool hasWd = weightDecay > 0f;
        int vocabSize = paramSpan.Length / embeddingDim;

        double pNormSq = 0.0;
        for (int i = 0; i < paramSpan.Length; i++) pNormSq += (double)paramSpan[i] * paramSpan[i];
        double pNorm = Math.Sqrt(pNormSq);
        double gNormSq = 0.0;
        foreach (var sparseObj in sparseList)
        {
            var sparse = (SparseEmbeddingGradient<float>)(object)sparseObj;
            int n = sparse.NumIndices * embeddingDim;
            var vs = sparse.Values.Data.Span;
            for (int k = 0; k < n; k++) gNormSq += (double)vs[k] * vs[k];
        }
        double gNorm = Math.Sqrt(gNormSq);
        double localLrD;
        if (pNorm < eps || gNorm < eps) localLrD = lr;
        else localLrD = lr * trustCoeff * pNorm / (gNorm + weightDecay * pNorm + eps);
        float localLr = (float)localLrD;

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
                    float vNew = momentum * velSpan[paramBase + c] + localLr * g;
                    velSpan[paramBase + c] = vNew;
                    paramSpan[paramBase + c] = theta - vNew;
                }
            }
        }
    }
}
