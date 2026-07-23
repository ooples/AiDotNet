using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Optimizers;

/// <summary>
/// Sparse scatter helper for Lion (Chen et al., 2023 — EvoLved Sign Momentum).
/// Update: <c>c = β1·m + (1-β1)·g; θ -= lr·(sign(c) + wd·θ); m ← β2·m + (1-β2)·g</c>.
/// </summary>
internal static partial class SparseEmbeddingOptimizerHelpers
{
    internal static bool TryApplyLionSparse<T>(
        Tensor<T> param, Tensor<T> m,
        double lr, double b1, double b2, double weightDecay)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (m is null) throw new ArgumentNullException(nameof(m));

        var sparseList = DifferentiableOps.GetSparseEmbeddingGradsFor(param);
        if (sparseList is null || sparseList.Count == 0) return false;
        // Bail: per-chunk processing changes sign(cEff) and the momentum state
        // for what should be one optimizer step on the summed gradient; non-zero
        // weight decay would also skip the decay update on untouched rows.
        if (sparseList.Count != 1) return false;
        if (weightDecay != 0.0) return false;
        if (HasDuplicateRows(sparseList)) return false;
        if (param.Rank != 2 || m.Rank != 2) return false;
        int vocabSize = param.Shape[0];
        int embeddingDim = param.Shape[1];
        if (m.Shape[0] != vocabSize || m.Shape[1] != embeddingDim) return false;

        double oneMinusB1 = 1.0 - b1;
        double oneMinusB2 = 1.0 - b2;

        if (typeof(T) == typeof(double))
        {
            ApplyLionSparseDouble(param, m, sparseList, embeddingDim,
                lr, b1, b2, oneMinusB1, oneMinusB2, weightDecay);
            return true;
        }
        if (typeof(T) == typeof(float))
        {
            ApplyLionSparseFloat(param, m, sparseList, embeddingDim,
                (float)lr, (float)b1, (float)b2, (float)oneMinusB1, (float)oneMinusB2, (float)weightDecay);
            return true;
        }

        var ops = MathHelper.GetNumericOperations<T>();
        T lrT = ops.FromDouble(lr);
        T b1T = ops.FromDouble(b1), b2T = ops.FromDouble(b2);
        T omB1 = ops.FromDouble(oneMinusB1), omB2 = ops.FromDouble(oneMinusB2);
        T wdT = ops.FromDouble(weightDecay);
        T zero = ops.Zero, one = ops.One, negOne = ops.Negate(one);
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
                    T mCur = m[paramBase + c];
                    T cEff = ops.Add(ops.Multiply(b1T, mCur), ops.Multiply(omB1, g));
                    T sign = ops.GreaterThan(cEff, zero) ? one : (ops.LessThan(cEff, zero) ? negOne : zero);
                    T upd = hasWd ? ops.Add(sign, ops.Multiply(wdT, param[paramBase + c])) : sign;
                    param[paramBase + c] = ops.Subtract(param[paramBase + c], ops.Multiply(lrT, upd));
                    m[paramBase + c] = ops.Add(ops.Multiply(b2T, mCur), ops.Multiply(omB2, g));
                }
            }
        }
        return true;
    }

    private static void ApplyLionSparseDouble<T>(
        Tensor<T> param, Tensor<T> m,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        double lr, double b1, double b2, double oneMinusB1, double oneMinusB2, double weightDecay)
    {
        var paramSpan = ((Tensor<double>)(object)param).Data.Span;
        var mSpan = ((Tensor<double>)(object)m).Data.Span;
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
                    double mCur = mSpan[paramBase + c];
                    double cEff = b1 * mCur + oneMinusB1 * g;
                    double sign = cEff > 0.0 ? 1.0 : (cEff < 0.0 ? -1.0 : 0.0);
                    double upd = hasWd ? sign + weightDecay * paramSpan[paramBase + c] : sign;
                    paramSpan[paramBase + c] -= lr * upd;
                    mSpan[paramBase + c] = b2 * mCur + oneMinusB2 * g;
                }
            }
        }
    }

    private static void ApplyLionSparseFloat<T>(
        Tensor<T> param, Tensor<T> m,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        float lr, float b1, float b2, float oneMinusB1, float oneMinusB2, float weightDecay)
    {
        var paramSpan = ((Tensor<float>)(object)param).Data.Span;
        var mSpan = ((Tensor<float>)(object)m).Data.Span;
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
                    float mCur = mSpan[paramBase + c];
                    float cEff = b1 * mCur + oneMinusB1 * g;
                    float sign = cEff > 0f ? 1f : (cEff < 0f ? -1f : 0f);
                    float upd = hasWd ? sign + weightDecay * paramSpan[paramBase + c] : sign;
                    paramSpan[paramBase + c] -= lr * upd;
                    mSpan[paramBase + c] = b2 * mCur + oneMinusB2 * g;
                }
            }
        }
    }
}
