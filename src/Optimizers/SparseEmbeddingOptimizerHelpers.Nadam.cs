using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Optimizers;

/// <summary>
/// Sparse scatter helper for Nadam — Adam with Nesterov-style look-ahead.
/// The corrected first-moment is <c>mHat = (β1·m_new + (1-β1)·g) / bc1</c>;
/// otherwise the math matches Adam.
/// </summary>
internal static partial class SparseEmbeddingOptimizerHelpers
{
    internal static bool TryApplyNadamSparse<T>(
        Tensor<T> param, Tensor<T> m, Tensor<T> v,
        double lr, double b1, double b2, double bc1, double bc2, double eps, double weightDecay)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));

        var sparseList = DifferentiableOps.GetSparseEmbeddingGradsFor(param);
        if (sparseList is null || sparseList.Count == 0) return false;
        // Bail: multi-chunk advances m/v once per chunk instead of on the summed
        // gradient (Nesterov correction also diverges); non-zero weight decay
        // would skip untouched rows in dense NAdam.
        if (sparseList.Count != 1) return false;
        if (weightDecay != 0.0) return false;
        if (HasDuplicateRows(sparseList)) return false;
        if (param.Rank != 2 || m.Rank != 2 || v.Rank != 2) return false;
        int vocabSize = param.Shape[0];
        int embeddingDim = param.Shape[1];
        if (m.Shape[0] != vocabSize || m.Shape[1] != embeddingDim) return false;
        if (v.Shape[0] != vocabSize || v.Shape[1] != embeddingDim) return false;

        double oneMinusB1 = 1.0 - b1;
        double oneMinusB2 = 1.0 - b2;

        if (typeof(T) == typeof(double))
        {
            ApplyNadamSparseDouble(param, m, v, sparseList, embeddingDim,
                lr, b1, b2, oneMinusB1, oneMinusB2, bc1, bc2, eps, weightDecay);
            return true;
        }
        if (typeof(T) == typeof(float))
        {
            ApplyNadamSparseFloat(param, m, v, sparseList, embeddingDim,
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
                    m[paramBase + c] = mNew;
                    v[paramBase + c] = vNew;
                    // Nesterov-corrected: mHat = (β1·mNew + (1-β1)·g) / bc1.
                    T mHat = ops.Divide(ops.Add(ops.Multiply(b1T, mNew), ops.Multiply(omB1, g)), bc1T);
                    T vHat = ops.Divide(vNew, bc2T);
                    T denom = ops.Add(ops.Sqrt(vHat), epsT);
                    param[paramBase + c] = ops.Subtract(theta, ops.Divide(ops.Multiply(lrT, mHat), denom));
                }
            }
        }
        return true;
    }

    private static void ApplyNadamSparseDouble<T>(
        Tensor<T> param, Tensor<T> m, Tensor<T> v,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        double lr, double b1, double b2, double oneMinusB1, double oneMinusB2,
        double bc1, double bc2, double eps, double weightDecay)
    {
        var paramSpan = ((Tensor<double>)(object)param).Data.Span;
        var mSpan = ((Tensor<double>)(object)m).Data.Span;
        var vSpan = ((Tensor<double>)(object)v).Data.Span;
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
                    mSpan[paramBase + c] = mNew;
                    vSpan[paramBase + c] = vNew;
                    double mHat = (b1 * mNew + oneMinusB1 * g) / bc1;
                    double vHat = vNew / bc2;
                    paramSpan[paramBase + c] = theta - lr * mHat / (Math.Sqrt(vHat) + eps);
                }
            }
        }
    }

    private static void ApplyNadamSparseFloat<T>(
        Tensor<T> param, Tensor<T> m, Tensor<T> v,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        float lr, float b1, float b2, float oneMinusB1, float oneMinusB2,
        float bc1, float bc2, float eps, float weightDecay)
    {
        var paramSpan = ((Tensor<float>)(object)param).Data.Span;
        var mSpan = ((Tensor<float>)(object)m).Data.Span;
        var vSpan = ((Tensor<float>)(object)v).Data.Span;
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
                    mSpan[paramBase + c] = mNew;
                    vSpan[paramBase + c] = vNew;
                    float mHat = (b1 * mNew + oneMinusB1 * g) / bc1;
                    float vHat = vNew / bc2;
                    paramSpan[paramBase + c] = theta - lr * mHat / ((float)Math.Sqrt(vHat) + eps);
                }
            }
        }
    }
}
