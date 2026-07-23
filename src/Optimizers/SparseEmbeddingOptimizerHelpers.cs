using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Optimizers;

/// <summary>
/// Sparse-aware optimizer fast paths for embedding-table parameters whose
/// backward came from <see cref="SparseEmbeddingGradient{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// An embedding-lookup forward gathers <c>numIndices</c> rows out of a
/// <c>[vocabSize, embeddingDim]</c> table — typically a 16-token sequence into
/// a 250 002-vocab BERT/XLM-R table. The dense backward materialises a
/// <c>[vocabSize, embeddingDim]</c> gradient that's overwhelmingly zero, and a
/// naive Adam step then reads + writes <c>m</c>, <c>v</c>, <c>θ</c> over every
/// row including the 249 986 that stayed zero. For paper-default LayoutXLM
/// that's ~192 M cells of m + v + θ memory traffic per step against ~16 rows
/// of real signal.
/// </para>
/// <para>
/// AiDotNet.Tensors#553 wires the sparse representation through the tape
/// alongside the existing dense seeding. Sparse-aware optimizers
/// (<see cref="AdamOptimizer{T, TInput, TOutput}"/>,
/// <see cref="AdamWOptimizer{T, TInput, TOutput}"/>) call
/// <see cref="TryApplyAdamSparse{T}"/> before the dense path; on a hit they
/// update <c>m</c>, <c>v</c>, and <c>θ</c> ONLY on the indexed rows and skip
/// the dense traversal entirely. The other 13+ optimizers in the tree keep
/// reading the dense grad as before — Tensors PR #553's backward seeds both
/// representations specifically so the rollout stays backward-compatible.
/// </para>
/// <para>
/// Duplicate indices in a single sparse contribution are NOT pre-aggregated:
/// scatter-add semantics REQUIRE that a row gathered N times in the forward
/// receives the SUM of N gradient rows on the backward. The implementation
/// here walks indices in order and accumulates Adam's <c>m</c> / <c>v</c>
/// moments and the parameter step incrementally per access, matching the
/// dense reference's behavior for non-unique index sets.
/// </para>
/// </remarks>
internal static partial class SparseEmbeddingOptimizerHelpers
{
    /// <summary>
    /// Walks every parameter on the context and, for any whose dense gradient
    /// is missing from <see cref="AiDotNet.Tensors.Engines.Autodiff.TapeStepContext{T}.Gradients"/>
    /// but has a sparse contribution recorded on
    /// <see cref="DifferentiableOps"/>, materialises the sparse list via
    /// <see cref="SparseEmbeddingGradient{T}.ToDense(IEngine)"/> and inserts the
    /// dense result back into <c>context.Gradients</c> so downstream code paths
    /// that consume a flat gradient vector (BFGS, LBFGS, Newton, TrustRegion,
    /// LevenbergMarquardt, ConjugateGradient, CoordinateDescent, ADMM, DFP —
    /// every second-order / line-search / quasi-Newton optimizer) see the
    /// embedding gradients alongside everything else. Must be called at the
    /// top of <c>Step(TapeStepContext)</c> before
    /// <c>GetFlatGradients()</c> / Hessian assembly — by then it's too late
    /// to backfill.
    /// </summary>
    /// <remarks>
    /// <para>
    /// No-op (zero allocs, single dictionary lookup) for every non-embedding
    /// parameter — the dense entry is already present and the sparse list is
    /// null. Embedding parameters that already have a dense entry (today's
    /// Tensors-side dense-seeding) also no-op. The materialisation only
    /// allocates when the Tensors-side dense seeding has been disabled and
    /// the embedding param's gradient lives only in the sparse list (the
    /// sparse-only future state).
    /// </para>
    /// </remarks>
    internal static void MaterializeSparseIntoGradientsDict<T>(
        AiDotNet.Tensors.Engines.Autodiff.TapeStepContext<T> context,
        IEngine engine)
    {
        if (context is null) throw new System.ArgumentNullException(nameof(context));

        foreach (var param in context.Parameters)
        {
            if (param is null) continue;
            // ContainsKey isn't enough: the autodiff path can leave a NULL placeholder
            // under the parameter key when a gradient wasn't computed. TryGetValue +
            // null check treats those entries as missing so the sparse contribution
            // still materializes — without this, flat-gradient optimizers (BFGS /
            // LBFGS / Newton / TrustRegion) calling GetFlatGradients() would silently
            // omit the embedding param's update.
            if (context.Gradients.TryGetValue(param, out var existing) && existing is not null)
                continue;

            var sparseList = DifferentiableOps.GetSparseEmbeddingGradsFor(param);
            if (sparseList is null || sparseList.Count == 0) continue;

            var materialised = sparseList[0].ToDense(engine);
            for (int i = 1; i < sparseList.Count; i++)
            {
                var next = sparseList[i].ToDense(engine);
                materialised = engine.TensorAdd(materialised, next);
            }
            context.Gradients[param] = materialised;
        }
    }

    /// <summary>
    /// Sparse-by-default gradient lookup for the dense-path optimizers. Reads
    /// <paramref name="context"/>'s gradient dict; when there's no dense entry
    /// but a sparse contribution exists (the future state once Tensors stops
    /// seeding the dense [vocabSize, embeddingDim] alloc alongside the sparse
    /// hint), materialises the sparse list via
    /// <see cref="SparseEmbeddingGradient{T}.ToDense(IEngine)"/> and returns
    /// the dense tensor. Multiple sparse contributions accumulate (scatter-add)
    /// via repeated ToDense calls combined with element-wise add.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Today's Tensors backward still seeds dense alongside sparse for
    /// backward-compat with the 17 in-tree dense-path optimizers, so this
    /// helper's fast path is "context.Gradients had the dense, return it
    /// unchanged" — bit-identical to the pre-refactor behaviour. The sparse
    /// fallback engages only when dense seeding is disabled (future Tensors
    /// PR) or when a Tape's GradientTape ran with create_graph + a custom
    /// op that contributed only to the sparse dict.
    /// </para>
    /// <para>
    /// <b>Why all dense-path optimizers route through this:</b> sparse-by-
    /// default means an embedding parameter's gradient lives in the sparse
    /// dict by default, and dense-only optimizers (SGD, Momentum, RMSprop,
    /// Adagrad, Adadelta, Lion, Sophia, Nadam, RAdam, LAMB, LARS, FTRL,
    /// AdaMax, AMSGrad, Adam8Bit, ProximalGradientDescent) call ToDense
    /// internally rather than relying on Tensors to scatter for them.
    /// Adam and AdamW skip this and take the scatter fast path via
    /// <see cref="TryApplyAdamSparse{T}"/>.
    /// </para>
    /// </remarks>
    /// <returns><c>true</c> when an effective gradient was resolved (dense in
    /// context, OR materialised from sparse), <c>false</c> when the param has
    /// no gradient contribution this step (the caller should <c>continue</c>).</returns>
    /// <summary>
    /// Cheap presence check for a parameter's sparse-embedding gradient — a dictionary
    /// lookup, NO sparse→dense materialization. Lets a caller decide whether to skip a
    /// parameter (and attempt the sparse fast path) BEFORE paying the full-tensor
    /// <c>ToDense</c> cost that <see cref="TryGetEffectiveGradient{T}"/> incurs when only
    /// sparse grads exist.
    /// </summary>
    internal static bool HasSparseEmbeddingGrad<T>(Tensor<T> param)
    {
        var list = DifferentiableOps.GetSparseEmbeddingGradsFor(param);
        return list is not null && list.Count > 0;
    }

    /// <summary>
    /// Returns true when any row index appears more than once in the sparse payload.
    /// Adaptive optimizers (Adam / AMSGrad / RMSProp / Adagrad / AdaDelta / NAdam /
    /// Adamax / FTRL / Lion / LAMB / LARS / Adam8Bit / ProximalL1) all assume the
    /// gradient for a row has been scatter-add coalesced before the optimizer step
    /// runs. Applying the per-occurrence helper twice for the same row corrupts
    /// moment / accumulator / sign state. Plain SGD without momentum is the only
    /// helper for which duplicates are safe (the update is linear in the gradient).
    /// Every non-SGD helper bails to dense materialization when this returns true.
    /// </summary>
    internal static bool HasDuplicateRows<T>(IReadOnlyList<SparseEmbeddingGradient<T>> sparseList)
    {
        if (sparseList is null) return false;
        var seen = new System.Collections.Generic.HashSet<long>();
        foreach (var sparse in sparseList)
        {
            int n = sparse.NumIndices;
            for (int k = 0; k < n; k++)
            {
                long row = sparse.Indices[k];
                if (!seen.Add(row)) return true;
            }
        }
        return false;
    }

    internal static bool TryGetEffectiveGradient<T>(
        AiDotNet.Tensors.Engines.Autodiff.TapeStepContext<T> context,
        Tensor<T> param,
        IEngine engine,
        out Tensor<T> grad)
    {
        if (context is null) throw new System.ArgumentNullException(nameof(context));
        if (param is null) throw new System.ArgumentNullException(nameof(param));

        if (context.Gradients.TryGetValue(param, out var denseGrad) && denseGrad is not null)
        {
            grad = denseGrad;
            return true;
        }

        var sparseList = DifferentiableOps.GetSparseEmbeddingGradsFor(param);
        if (sparseList is null || sparseList.Count == 0)
        {
            grad = null!;
            return false;
        }

        // Materialise the sparse list to dense — ToDense applies the scatter-add
        // semantics (duplicate indices accumulate), and multiple sparse entries
        // for the same param sum via the engine's tensor add.
        var materialised = sparseList[0].ToDense(engine);
        for (int i = 1; i < sparseList.Count; i++)
        {
            var next = sparseList[i].ToDense(engine);
            materialised = engine.TensorAdd(materialised, next);
        }
        grad = materialised;
        return true;
    }


    /// <summary>
    /// Attempts an Adam (or AdamW when <paramref name="weightDecay"/> &gt; 0)
    /// update on a sparse-embedding parameter by scattering moment + weight
    /// updates over only the accessed rows. Returns <c>false</c> when the
    /// parameter has no sparse contributions on this backward (the caller
    /// should fall back to the dense path).
    /// </summary>
    /// <typeparam name="T">Parameter element type — <c>double</c> and
    /// <c>float</c> take a raw-array fast path; other Ts go through
    /// <see cref="MathHelper"/>'s generic <c>INumericOperations&lt;T&gt;</c>.</typeparam>
    /// <param name="param">The trainable parameter — must have rank 2 with shape
    /// <c>[vocabSize, embeddingDim]</c> matching what
    /// <see cref="SparseEmbeddingGradient{T}.VocabSize"/> /
    /// <see cref="SparseEmbeddingGradient{T}.EmbeddingDim"/> reported.</param>
    /// <param name="m">Adam first-moment buffer aligned to <paramref name="param"/>.</param>
    /// <param name="v">Adam second-moment buffer aligned to <paramref name="param"/>.</param>
    /// <param name="lr">Effective learning rate (post-schedule) for this step.</param>
    /// <param name="b1">Adam beta1.</param>
    /// <param name="b2">Adam beta2.</param>
    /// <param name="bc1">Adam bias-correction-1 = 1 − β1^step.</param>
    /// <param name="bc2">Adam bias-correction-2 = 1 − β2^step.</param>
    /// <param name="eps">Adam epsilon (denominator floor).</param>
    /// <param name="weightDecay">AdamW decoupled weight-decay rate. Pass 0 for plain Adam.</param>
    /// <returns><c>true</c> when a sparse update was applied; <c>false</c> when the
    /// caller must run the dense Adam path.</returns>
    internal static bool TryApplyAdamSparse<T>(
        Tensor<T> param,
        Tensor<T> m,
        Tensor<T> v,
        double lr,
        double b1,
        double b2,
        double bc1,
        double bc2,
        double eps,
        double weightDecay = 0.0)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));

        var sparseList = DifferentiableOps.GetSparseEmbeddingGradsFor(param);
        if (sparseList is null || sparseList.Count == 0) return false;
        // Bail for multi-chunk: m, v would advance once per sparse contribution
        // instead of once on the summed gradient. The dense path handles this
        // correctly via ToDense materialization in TryGetEffectiveGradient.
        if (sparseList.Count != 1) return false;
        // Bail when the sparse payload has duplicate rows: scatter-add would
        // sum them before the optimizer step in dense; per-occurrence here
        // applies Adam twice for the same row and corrupts m/v.
        if (HasDuplicateRows(sparseList)) return false;

        // Embedding tables are rank-2 [vocabSize, embeddingDim]. The sparse-grad
        // factory enforces matching VocabSize/EmbeddingDim against the param's
        // dimensions, but the in-place path here can only walk a contiguous
        // rank-2 buffer — bail to the dense path on any other layout (e.g.
        // a downstream consumer wrapped the table in extra batch dims).
        if (param.Rank != 2 || m.Rank != 2 || v.Rank != 2) return false;
        int vocabSize = param.Shape[0];
        int embeddingDim = param.Shape[1];
        if (m.Shape[0] != vocabSize || m.Shape[1] != embeddingDim) return false;
        if (v.Shape[0] != vocabSize || v.Shape[1] != embeddingDim) return false;

        double oneMinusB1 = 1.0 - b1;
        double oneMinusB2 = 1.0 - b2;
        bool isDouble = typeof(T) == typeof(double);
        bool isFloat = typeof(T) == typeof(float);

        if (isDouble)
        {
            ApplyAdamSparseDouble(
                param, m, v,
                sparseList, embeddingDim,
                lr, b1, b2, oneMinusB1, oneMinusB2, bc1, bc2, eps, weightDecay);
            return true;
        }
        if (isFloat)
        {
            ApplyAdamSparseFloat(
                param, m, v,
                sparseList, embeddingDim,
                (float)lr, (float)b1, (float)b2,
                (float)oneMinusB1, (float)oneMinusB2,
                (float)bc1, (float)bc2, (float)eps, (float)weightDecay);
            return true;
        }

        // Generic-T fallback. Same per-row scatter, just through NumOps so
        // half / bfloat16 / decimal cores still get the sparse savings.
        var ops = MathHelper.GetNumericOperations<T>();
        T lrT = ops.FromDouble(lr);
        T b1T = ops.FromDouble(b1);
        T b2T = ops.FromDouble(b2);
        T oneMinusB1T = ops.FromDouble(oneMinusB1);
        T oneMinusB2T = ops.FromDouble(oneMinusB2);
        T bc1T = ops.FromDouble(bc1);
        T bc2T = ops.FromDouble(bc2);
        T epsT = ops.FromDouble(eps);
        T wdT = ops.FromDouble(weightDecay);
        T oneT = ops.One;
        bool hasWd = weightDecay > 0.0;

        foreach (var sparse in sparseList)
        {
            int numIndices = sparse.NumIndices;
            if (numIndices == 0) continue;
            var values = sparse.Values;        // [numIndices, embeddingDim]
            var indices = sparse.Indices;      // [numIndices], dtype long
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
                    T vCur = v[paramBase + c];
                    // m = β1·m + (1−β1)·g
                    T mNew = ops.Add(ops.Multiply(b1T, mCur), ops.Multiply(oneMinusB1T, g));
                    // v = β2·v + (1−β2)·g²
                    T gSq = ops.Multiply(g, g);
                    T vNew = ops.Add(ops.Multiply(b2T, vCur), ops.Multiply(oneMinusB2T, gSq));
                    m[paramBase + c] = mNew;
                    v[paramBase + c] = vNew;
                    T mHat = ops.Divide(mNew, bc1T);
                    T vHat = ops.Divide(vNew, bc2T);
                    T denom = ops.Add(ops.Sqrt(vHat), epsT);
                    T step = ops.Multiply(lrT, ops.Divide(mHat, denom));
                    T theta = param[paramBase + c];
                    if (hasWd)
                    {
                        // AdamW: decoupled weight decay θ ← θ(1 − lr·wd) − step
                        theta = ops.Multiply(theta, ops.Subtract(oneT, ops.Multiply(lrT, wdT)));
                    }
                    param[paramBase + c] = ops.Subtract(theta, step);
                }
            }
        }
        return true;
    }

    private static void ApplyAdamSparseDouble<T>(
        Tensor<T> param,
        Tensor<T> m,
        Tensor<T> v,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList,
        int embeddingDim,
        double lr, double b1, double b2,
        double oneMinusB1, double oneMinusB2,
        double bc1, double bc2, double eps, double weightDecay)
    {
        // Get writable spans into the underlying buffers (param/m/v are the
        // SAME tensors registered with the optimizer; their .Data buffers are
        // what subsequent Forwards read, so updating in place is required).
        var paramSpan = ((Tensor<double>)(object)param).Data.Span;
        var mSpan = ((Tensor<double>)(object)m).Data.Span;
        var vSpan = ((Tensor<double>)(object)v).Data.Span;
        bool hasWd = weightDecay > 0.0;
        double wdDecayFactor = hasWd ? (1.0 - lr * weightDecay) : 1.0;

        foreach (var sparseObj in sparseList)
        {
            // sparseObj is SparseEmbeddingGradient<T>; we know T=double here.
            var sparse = (SparseEmbeddingGradient<double>)(object)sparseObj;
            int numIndices = sparse.NumIndices;
            if (numIndices == 0) continue;
            var valuesSpan = sparse.Values.Data.Span;
            int vocabSize = paramSpan.Length / embeddingDim;
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
                    double vCur = vSpan[paramBase + c];
                    double mNew = b1 * mCur + oneMinusB1 * g;
                    double vNew = b2 * vCur + oneMinusB2 * g * g;
                    mSpan[paramBase + c] = mNew;
                    vSpan[paramBase + c] = vNew;
                    double mHat = mNew / bc1;
                    double vHat = vNew / bc2;
                    double step = lr * mHat / (Math.Sqrt(vHat) + eps);
                    double theta = paramSpan[paramBase + c];
                    if (hasWd) theta *= wdDecayFactor;
                    paramSpan[paramBase + c] = theta - step;
                }
            }
        }
    }

    private static void ApplyAdamSparseFloat<T>(
        Tensor<T> param,
        Tensor<T> m,
        Tensor<T> v,
        IReadOnlyList<SparseEmbeddingGradient<T>> sparseList,
        int embeddingDim,
        float lr, float b1, float b2,
        float oneMinusB1, float oneMinusB2,
        float bc1, float bc2, float eps, float weightDecay)
    {
        var paramSpan = ((Tensor<float>)(object)param).Data.Span;
        var mSpan = ((Tensor<float>)(object)m).Data.Span;
        var vSpan = ((Tensor<float>)(object)v).Data.Span;
        bool hasWd = weightDecay > 0f;
        float wdDecayFactor = hasWd ? (1f - lr * weightDecay) : 1f;

        foreach (var sparseObj in sparseList)
        {
            var sparse = (SparseEmbeddingGradient<float>)(object)sparseObj;
            int numIndices = sparse.NumIndices;
            if (numIndices == 0) continue;
            var valuesSpan = sparse.Values.Data.Span;
            int vocabSize = paramSpan.Length / embeddingDim;
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
                    float vCur = vSpan[paramBase + c];
                    float mNew = b1 * mCur + oneMinusB1 * g;
                    float vNew = b2 * vCur + oneMinusB2 * g * g;
                    mSpan[paramBase + c] = mNew;
                    vSpan[paramBase + c] = vNew;
                    float mHat = mNew / bc1;
                    float vHat = vNew / bc2;
                    float step = lr * mHat / ((float)Math.Sqrt(vHat) + eps);
                    float theta = paramSpan[paramBase + c];
                    if (hasWd) theta *= wdDecayFactor;
                    paramSpan[paramBase + c] = theta - step;
                }
            }
        }
    }

}
