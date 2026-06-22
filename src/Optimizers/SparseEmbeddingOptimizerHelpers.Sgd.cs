using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Optimizers;

/// <summary>
/// Sparse scatter helpers for the SGD optimizer family — plain SGD,
/// SGD-with-momentum, and (AiDotNet-flavored) Nesterov-Accelerated SGD.
/// </summary>
/// <remarks>
/// <para><b>State-schema note.</b> AiDotNet's <c>MomentumOptimizer</c> and
/// <c>NesterovAcceleratedGradientOptimizer</c> use the <i>learning-rate-scaled
/// velocity</i> convention:
/// <code>
///   v ← momentum · v + lr · g
///   θ ← θ − v
/// </code>
/// This differs from PyTorch's convention (<c>v ← momentum·v + g; θ ← θ − lr·v</c>)
/// — the two are mathematically equivalent but the buffer values differ by a
/// factor of <c>lr</c>, so mixing sparse and dense steps would corrupt the
/// state. The sparse helper below matches AiDotNet's convention exactly so
/// you can interleave sparse + dense steps within one training run without
/// the velocity buffer drifting.</para>
/// <para>Has three implementations: a raw-array fast path for <c>T = double</c>,
/// the same for <c>T = float</c>, and a generic
/// <see cref="INumericOperations{T}"/> fallback.</para>
/// </remarks>
internal static partial class SparseEmbeddingOptimizerHelpers
{
    /// <summary>
    /// Sparse SGD with optional momentum (AiDotNet's lr-scaled velocity schema).
    /// Pass <paramref name="velocity"/> = null and <paramref name="momentum"/> = 0
    /// for plain SGD (no state buffer touched).
    /// </summary>
    /// <returns>true on sparse-path hit; false → caller must run dense.</returns>
    internal static bool TryApplySgdSparse<T>(
        Tensor<T> param, Tensor<T>? velocity,
        double lr, double momentum, double weightDecay)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));

        var sparseList = DifferentiableOps.GetSparseEmbeddingGradsFor(param);
        if (sparseList is null || sparseList.Count == 0) return false;

        // Bail to dense for cases where sparse semantics diverge from dense SGD:
        //   * Multiple sparse contributions per param — applying the optimizer
        //     once per chunk is not the same as applying it once on the summed
        //     gradient (only true equivalence is for the plain-SGD additive
        //     `θ -= lr·g` case below, and the helper would still update velocity
        //     wrong on momentum paths).
        //   * Non-zero momentum — dense applies `v = momentum·v + lr·g; θ -= v`
        //     across every index; sparse would skip untouched velocity entries,
        //     so the momentum buffer drifts.
        //   * Non-zero weight decay — dense applies `g += wd·θ` to every index;
        //     sparse only sees touched rows.
        // Plain SGD with momentum == 0 and weightDecay == 0 IS safe: the update
        // `θ -= lr·g` is additive and per-element-independent, so touching only
        // the indices that received gradient is exactly equivalent to the dense
        // step (untouched indices have g=0 and would not change anyway).
        if (sparseList.Count != 1) return false;
        if (momentum != 0.0 || weightDecay != 0.0) return false;
        // Plain SGD's `θ -= lr·g` is linear in g, so applying the update twice for
        // the same row is mathematically identical to applying it once on the sum.
        // We still bail on duplicates because the rest of the helper family does the
        // same, keeping the caller's behavior uniform: any duplicate-row sparse grad
        // routes through ToDense regardless of which optimizer is configured.
        if (HasDuplicateRows(sparseList)) return false;

        if (param.Rank != 2) return false;
        int vocabSize = param.Shape[0];
        int embeddingDim = param.Shape[1];
        if (velocity is not null)
        {
            if (velocity.Rank != 2 || velocity.Shape[0] != vocabSize || velocity.Shape[1] != embeddingDim)
                return false;
        }

        bool hasMomentum = velocity is not null && momentum != 0.0;

        if (typeof(T) == typeof(double))
        {
            ApplySgdSparseDouble(param, velocity, sparseList, embeddingDim,
                lr, momentum, weightDecay, hasMomentum);
            return true;
        }
        if (typeof(T) == typeof(float))
        {
            ApplySgdSparseFloat(param, velocity, sparseList, embeddingDim,
                (float)lr, (float)momentum, (float)weightDecay, hasMomentum);
            return true;
        }

        // Generic-T fallback.
        var ops = MathHelper.GetNumericOperations<T>();
        T lrT = ops.FromDouble(lr);
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
                    if (hasMomentum)
                    {
                        // AiDotNet schema: v ← momentum · v + lr · g; θ ← θ − v.
                        T vOld = velocity![paramBase + c];
                        T vNew = ops.Add(ops.Multiply(momT, vOld), ops.Multiply(lrT, g));
                        velocity[paramBase + c] = vNew;
                        param[paramBase + c] = ops.Subtract(theta, vNew);
                    }
                    else
                    {
                        // Plain SGD: θ ← θ − lr · g.
                        param[paramBase + c] = ops.Subtract(theta, ops.Multiply(lrT, g));
                    }
                }
            }
        }
        return true;
    }

    private static void ApplySgdSparseDouble<T>(
        Tensor<T> param, Tensor<T>? velocity,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        double lr, double momentum, double weightDecay, bool hasMomentum)
    {
        var paramSpan = ((Tensor<double>)(object)param).Data.Span;
        var velSpan = hasMomentum ? ((Tensor<double>)(object)velocity!).Data.Span : default;
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
                    if (hasMomentum)
                    {
                        double vNew = momentum * velSpan[paramBase + c] + lr * g;
                        velSpan[paramBase + c] = vNew;
                        paramSpan[paramBase + c] = theta - vNew;
                    }
                    else
                    {
                        paramSpan[paramBase + c] = theta - lr * g;
                    }
                }
            }
        }
    }

    private static void ApplySgdSparseFloat<T>(
        Tensor<T> param, Tensor<T>? velocity,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList, int embeddingDim,
        float lr, float momentum, float weightDecay, bool hasMomentum)
    {
        var paramSpan = ((Tensor<float>)(object)param).Data.Span;
        var velSpan = hasMomentum ? ((Tensor<float>)(object)velocity!).Data.Span : default;
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
                    if (hasMomentum)
                    {
                        float vNew = momentum * velSpan[paramBase + c] + lr * g;
                        velSpan[paramBase + c] = vNew;
                        paramSpan[paramBase + c] = theta - vNew;
                    }
                    else
                    {
                        paramSpan[paramBase + c] = theta - lr * g;
                    }
                }
            }
        }
    }
}
