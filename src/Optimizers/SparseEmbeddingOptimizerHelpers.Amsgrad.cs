using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Optimizers;

/// <summary>
/// Sparse scatter helper for AMSGrad (Reddi et al., 2018). Like Adam but
/// uses <c>vMax = max(vMax, v_new)</c> in the denominator to preserve the
/// monotonic non-increasing learning-rate guarantee. The vMax max is taken
/// only at touched indices — untouched vMax stay at their previous values
/// (consistent with the sparse-grad semantics).
/// </summary>
internal static partial class SparseEmbeddingOptimizerHelpers
{
    internal static bool TryApplyAmsgradSparse<T>(
        Tensor<T> param, Tensor<T> m, Tensor<T> v, Tensor<T> vMax,
        double lr, double b1, double b2, double bc1, double bc2, double eps, double weightDecay)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (vMax is null) throw new ArgumentNullException(nameof(vMax));

        var sparseList = DifferentiableOps.GetSparseEmbeddingGradsFor(param);
        if (sparseList is null || sparseList.Count == 0) return false;
        // Bail: multi-chunk advances (m, v, vMax) once per chunk instead of on
        // the summed gradient; non-zero weight decay would skip untouched rows
        // in dense AMSGrad. Also the dense AMSGradOptimizer in this codebase does
        // NOT apply bias correction in the denominator path — callers passing
        // bias-corrected hyperparameters would diverge from dense; we keep the
        // bail-out broad rather than introducing two AMSGrad flavors.
        if (sparseList.Count != 1) return false;
        if (weightDecay != 0.0) return false;
        if (HasDuplicateRows(sparseList)) return false;
        if (param.Rank != 2 || m.Rank != 2 || v.Rank != 2 || vMax.Rank != 2) return false;
        int vocabSize = param.Shape[0];
        int embeddingDim = param.Shape[1];
        if (m.Shape[0] != vocabSize || m.Shape[1] != embeddingDim) return false;
        if (v.Shape[0] != vocabSize || v.Shape[1] != embeddingDim) return false;
        if (vMax.Shape[0] != vocabSize || vMax.Shape[1] != embeddingDim) return false;

        double oneMinusB1 = 1.0 - b1;
        double oneMinusB2 = 1.0 - b2;

        if (typeof(T) == typeof(double))
        {
            ApplyAmsgradSparseDouble(param, m, v, vMax, sparseList, embeddingDim,
                lr, b1, b2, oneMinusB1, oneMinusB2, bc1, bc2, eps, weightDecay);
            return true;
        }
        if (typeof(T) == typeof(float))
        {
            ApplyAmsgradSparseFloat(param, m, v, vMax, sparseList, embeddingDim,
                (float)lr, (float)b1, (float)b2, (float)oneMinusB1, (float)oneMinusB2,
                (float)bc1, (float)bc2, (float)eps, (float)weightDecay);
            return true;
        }

        var ops = MathHelper.GetNumericOperations<T>();
        T lrT = ops.FromDouble(lr);
        T b1T = ops.FromDouble(b1), b2T = ops.FromDouble(b2);
        T omB1 = ops.FromDouble(oneMinusB1), omB2 = ops.FromDouble(oneMinusB2);
        T bc1T = ops.FromDouble(bc1), bc2T = ops.FromDouble(bc2);
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
                    T vNew = ops.Add(ops.Multiply(b2T, v[paramBase + c]), ops.Multiply(omB2, ops.Multiply(g, g)));
                    T vmOld = vMax[paramBase + c];
                    T vmNew = ops.GreaterThan(vmOld, vNew) ? vmOld : vNew;
                    m[paramBase + c] = mNew;
                    v[paramBase + c] = vNew;
                    vMax[paramBase + c] = vmNew;
                    T mHat = ops.Divide(mNew, bc1T);
                    // Dense AMSGradOptimizer does NOT divide vMax by bc2 (see lines
                    // 333-340 of AMSGradOptimizer.cs: vHat = max(vHat, v); denom =
                    // sqrt(vHat) + eps). Match that — applying bc2 here would
                    // produce a different effective lr from the dense path.
                    param[paramBase + c] = ops.Subtract(theta,
                        ops.Divide(ops.Multiply(lrT, mHat), ops.Add(ops.Sqrt(vmNew), epsT)));
                }
            }
        }
        return true;
    }

    private static void ApplyAmsgradSparseDouble<T>(
        Tensor<T> param, Tensor<T> m, Tensor<T> v, Tensor<T> vMax,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        double lr, double b1, double b2, double oneMinusB1, double oneMinusB2,
        double bc1, double bc2, double eps, double weightDecay)
    {
        var paramSpan = ((Tensor<double>)(object)param).Data.Span;
        var mSpan = ((Tensor<double>)(object)m).Data.Span;
        var vSpan = ((Tensor<double>)(object)v).Data.Span;
        var vmSpan = ((Tensor<double>)(object)vMax).Data.Span;
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
                    double vNew = b2 * vSpan[paramBase + c] + oneMinusB2 * g * g;
                    double vmNew = Math.Max(vmSpan[paramBase + c], vNew);
                    mSpan[paramBase + c] = mNew;
                    vSpan[paramBase + c] = vNew;
                    vmSpan[paramBase + c] = vmNew;
                    double mHat = mNew / bc1;
                    // Dense AMSGrad doesn't apply bc2 to vMax — see AMSGradOptimizer.cs.
                    paramSpan[paramBase + c] = theta - lr * mHat / (Math.Sqrt(vmNew) + eps);
                }
            }
        }
    }

    private static void ApplyAmsgradSparseFloat<T>(
        Tensor<T> param, Tensor<T> m, Tensor<T> v, Tensor<T> vMax,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        float lr, float b1, float b2, float oneMinusB1, float oneMinusB2,
        float bc1, float bc2, float eps, float weightDecay)
    {
        var paramSpan = ((Tensor<float>)(object)param).Data.Span;
        var mSpan = ((Tensor<float>)(object)m).Data.Span;
        var vSpan = ((Tensor<float>)(object)v).Data.Span;
        var vmSpan = ((Tensor<float>)(object)vMax).Data.Span;
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
                    float vNew = b2 * vSpan[paramBase + c] + oneMinusB2 * g * g;
                    float vmNew = Math.Max(vmSpan[paramBase + c], vNew);
                    mSpan[paramBase + c] = mNew;
                    vSpan[paramBase + c] = vNew;
                    vmSpan[paramBase + c] = vmNew;
                    float mHat = mNew / bc1;
                    // Dense AMSGrad doesn't apply bc2 to vMax — see AMSGradOptimizer.cs.
                    paramSpan[paramBase + c] = theta - lr * mHat / ((float)Math.Sqrt(vmNew) + eps);
                }
            }
        }
    }
}
