using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Shared dispatcher for memory-bounded chunked Predict / Train calls.
/// Centralises the "chunk along axis 0 when the model is a tensor neural
/// network and the input is large; pass through unchanged for everything
/// else" decision that surfaces at every full-batch site audited in #1296
/// (optimizer evaluator, cross-validators, AutoML trial loop, NAS supernet
/// warmup, diffusion AutoML).
///
/// <para>
/// Exposes four capabilities beyond the basic per-call chunking that
/// closed the #1296 OOM family. PyTorch / TensorFlow ship none of these
/// out of the box (1, 3, 4 are user-rolled patterns; 2 is unique):
/// </para>
/// <list type="number">
///   <item><b>Adaptive OOM recovery</b> (<see cref="PredictAdaptive{T,TInput,TOutput}"/>,
///     <see cref="TrainAdaptive{T,TInput,TOutput}"/>) — catches
///     <see cref="OutOfMemoryException"/>, halves the chunk size, retries.
///     Auto-grows back up on successive successes via a per-model ratchet
///     held in a weak-reference table.</item>
///   <item><b>Stream-aggregation</b> (<see cref="PredictAndReduce{T,TInput,TOutput,TAccumulator}"/>) —
///     per-chunk reducer callback that never materialises the full
///     predictions tensor. Available for callers (e.g. user-defined
///     metric paths) that only need a scalar aggregate over predictions
///     and want to skip the concat 2× peak the plain
///     <see cref="NeuralNetworkBase{T}.PredictInBatches"/> incurs.
///     The <see cref="OptimizerBase{T,TInput,TOutput}"/> R² / SS<sub>res</sub>
///     path currently routes through <see cref="PredictMaybeBatched{T,TInput,TOutput}"/>
///     because its downstream consumers (PredictionStats, alignment
///     helpers) need the materialised prediction tensor; wiring R² to
///     this reducer is a follow-up tracked separately.</item>
///   <item><b>True gradient accumulation</b> (<see cref="GradientAccumulationMode.Accumulate"/>) —
///     delegates to <see cref="NeuralNetworkBase{T}.TrainWithGradientAccumulation"/>
///     so the optimizer fires exactly once with averaged chunk gradients,
///     preserving full-batch direction AND keeping Adam's m/v cadence at
///     one update per logical batch.</item>
///   <item><b>Memory-budget API</b> (<see cref="PredictWithMemoryBudget{T,TInput,TOutput}"/>,
///     <see cref="EstimateChunkSize{T}"/>) — caller specifies a budget
///     in bytes; the helper probes per-sample cost with a B=1 forward
///     and picks the chunk size that fits with safety margin.</item>
/// </list>
///
/// <para>
/// <b>Note on compile routing:</b> Earlier drafts of this PR routed
/// per-chunk forwards through <see cref="NeuralNetworkBase{T}.PredictCompiled"/>
/// when <see cref="AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions"/>
/// has compilation enabled, but that interacted with the optimizer's
/// epoch-loop tape allocations and exhausted the unmanaged-commit limit.
/// The current implementation deliberately stays on the eager
/// <see cref="NeuralNetworkBase{T}.Predict"/> path; the companion
/// <see cref="CompiledModelHost{T}.Predict"/> SetInputs-rebind fix in
/// this PR still benefits any caller that explicitly invokes
/// <c>PredictCompiled</c>.
/// </para>
/// </summary>
internal static class NeuralBatchHelper
{
    /// <summary>
    /// Default chunk size used when callers don't specify one.
    /// </summary>
    public const int DefaultBatchSize = 256;

    /// <summary>Lower bound the adaptive paths clamp to before rethrowing.</summary>
    public const int MinAdaptiveBatchSize = 1;

    private const int SuccessesBeforeGrow = 4;

    /// <summary>
    /// Fraction of the user's stated memory budget that
    /// <see cref="EstimateChunkSize{T}"/> actually targets. 70 % leaves
    /// headroom for transient autodiff allocations the B=1 probe doesn't
    /// capture.
    /// </summary>
    public const double MemoryBudgetSafetyFactor = 0.7;

    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, AdaptiveState> _adaptiveStates
        = new();

    private sealed class AdaptiveState
    {
        public int CurrentBatchSize;
        public int ConsecutiveSuccesses;
    }

    /// <summary>
    /// Modes for <see cref="TrainMaybeBatched{T,TInput,TOutput}"/> when it
    /// chunks an NN call.
    /// </summary>
    public enum GradientAccumulationMode
    {
        /// <summary>
        /// Each chunk fires an independent SGD step — <c>ceil(N / batchSize)</c>
        /// optimizer updates. Matches standard mini-batch SGD; changes the
        /// gradient trajectory vs a full-batch step.
        /// </summary>
        Independent = 0,

        /// <summary>
        /// All chunks contribute to a single accumulator; one optimizer
        /// step fires at the end with the averaged gradient. Preserves
        /// full-batch direction AND keeps Adam's m/v cadence at one
        /// update per logical batch.
        /// </summary>
        Accumulate = 1,
    }

    // ─────────────────────────────────────────────────────────────────
    // PredictMaybeBatched / TrainMaybeBatched (baseline)
    // ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Routes <c>model.Predict(X)</c> through
    /// <see cref="NeuralNetworkBase{T}.PredictInBatches"/> when the model
    /// is a tensor neural network and <paramref name="X"/> is a
    /// <see cref="Tensor{T}"/> with a leading axis larger than
    /// <paramref name="batchSize"/>. Output is element-equivalent (modulo
    /// matmul reduction order) to a single-shot <c>Predict</c>.
    ///
    /// <para><b>Adaptive OOM recovery is on by default.</b> If the
    /// chunked forward throws <see cref="OutOfMemoryException"/>, the
    /// helper halves <paramref name="batchSize"/>, forces a GC, and
    /// retries — repeating down to 1. The per-model adaptive ratchet
    /// persists across calls so subsequent calls start from the
    /// last-known-good size instead of probing from scratch. Pass
    /// <paramref name="disableAdaptiveRetry"/> = <see langword="true"/>
    /// to opt out (raw OOM propagation for diagnostics).</para>
    /// </summary>
    public static TOutput PredictMaybeBatched<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        TInput X,
        int batchSize = DefaultBatchSize,
        bool disableAdaptiveRetry = false,
        int minBatchSize = MinAdaptiveBatchSize)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        if (batchSize < 1) batchSize = 1;
        if (minBatchSize < 1) minBatchSize = 1;
        if (minBatchSize > batchSize) minBatchSize = batchSize;

        if (!(model is NeuralNetworkBase<T> nn
            && X is Tensor<T> xTensor
            && xTensor.Rank >= 1))
        {
            return model.Predict(X);
        }

        // The leading-axis-larger-than-batchSize check is necessary but
        // not sufficient to call a tensor "batched". A genuine single
        // sample with shape [seq, F] or even [features] where the leading
        // axis happens to exceed batchSize would otherwise be sliced
        // along the wrong axis. When the architecture's expected unbatched
        // rank matches the input rank, the tensor is unbatched — let the
        // base Predict path handle promotion.
        if (IsLikelyUnbatched(nn, xTensor))
        {
            return model.Predict(X);
        }

        if (disableAdaptiveRetry)
        {
            if (xTensor.Shape[0] > batchSize)
            {
                var chunked = nn.PredictInBatches(xTensor, batchSize);
                if (chunked is TOutput typed) return typed;
            }
            return model.Predict(X);
        }

        var state = GetOrCreateAdaptiveState(model, batchSize);
        while (true)
        {
            try
            {
                int effectiveBatchSize = System.Math.Min(state.CurrentBatchSize, batchSize);
                TOutput result;
                if (xTensor.Shape[0] <= effectiveBatchSize)
                {
                    var direct = nn.Predict(xTensor);
                    result = direct is TOutput dt ? dt : model.Predict(X);
                }
                else
                {
                    var chunked = nn.PredictInBatches(xTensor, effectiveBatchSize);
                    result = chunked is TOutput typed ? typed : model.Predict(X);
                }
                state.ConsecutiveSuccesses++;
                MaybeGrowBatch(state, batchSize);
                return result;
            }
            catch (OutOfMemoryException)
            {
                if (state.CurrentBatchSize <= minBatchSize) throw;
                state.CurrentBatchSize = System.Math.Max(minBatchSize, state.CurrentBatchSize / 2);
                state.ConsecutiveSuccesses = 0;
                System.GC.Collect();
                System.GC.WaitForPendingFinalizers();
                System.GC.Collect();
            }
        }
    }

    /// <summary>
    /// Routes <c>model.Train(X, Y)</c> through chunked mini-batched
    /// <see cref="NeuralNetworkBase{T}.Train"/> calls (or a single
    /// gradient-accumulated step under <see cref="GradientAccumulationMode.Accumulate"/>)
    /// when the model is a tensor neural network and the inputs share a
    /// large leading axis. Non-NN models pass through.
    ///
    /// <para><b>Adaptive OOM recovery is on by default</b> — see
    /// <see cref="PredictMaybeBatched{T,TInput,TOutput}"/> for the
    /// halve-and-retry contract. Pass <paramref name="disableAdaptiveRetry"/>
    /// = <see langword="true"/> to opt out.</para>
    /// </summary>
    public static void TrainMaybeBatched<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        TInput X,
        TOutput Y,
        int batchSize = DefaultBatchSize,
        GradientAccumulationMode mode = GradientAccumulationMode.Independent,
        bool disableAdaptiveRetry = false,
        int minBatchSize = MinAdaptiveBatchSize)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        if (batchSize < 1) batchSize = 1;
        if (minBatchSize < 1) minBatchSize = 1;
        if (minBatchSize > batchSize) minBatchSize = batchSize;

        if (!(model is NeuralNetworkBase<T> nn
            && X is Tensor<T> xTensor
            && Y is Tensor<T> yTensor
            && xTensor.Rank >= 1
            && yTensor.Rank >= 1
            && xTensor.Shape[0] == yTensor.Shape[0]))
        {
            model.Train(X, Y);
            return;
        }

        // Same unbatched-input guard as PredictMaybeBatched: rank-equal-
        // to-unbatched-expected means the leading axis is sequence/
        // feature, not batch — chunking along it would corrupt semantics.
        if (IsLikelyUnbatched(nn, xTensor))
        {
            nn.Train(xTensor, yTensor);
            return;
        }

        if (disableAdaptiveRetry)
        {
            if (xTensor.Shape[0] > batchSize)
            {
                if (mode == GradientAccumulationMode.Accumulate)
                {
                    nn.TrainWithGradientAccumulation(xTensor, yTensor, batchSize);
                }
                else
                {
                    TrainIndependentChunks(nn, xTensor, yTensor, batchSize);
                }
            }
            else
            {
                nn.Train(xTensor, yTensor);
            }
            return;
        }

        var state = GetOrCreateAdaptiveState(model, batchSize);
        while (true)
        {
            try
            {
                int effectiveBatchSize = System.Math.Min(state.CurrentBatchSize, batchSize);
                if (xTensor.Shape[0] <= effectiveBatchSize)
                {
                    nn.Train(xTensor, yTensor);
                }
                else if (mode == GradientAccumulationMode.Accumulate)
                {
                    nn.TrainWithGradientAccumulation(xTensor, yTensor, effectiveBatchSize);
                }
                else
                {
                    TrainIndependentChunks(nn, xTensor, yTensor, effectiveBatchSize);
                }
                state.ConsecutiveSuccesses++;
                MaybeGrowBatch(state, batchSize);
                return;
            }
            catch (OutOfMemoryException)
            {
                if (state.CurrentBatchSize <= minBatchSize) throw;
                state.CurrentBatchSize = System.Math.Max(minBatchSize, state.CurrentBatchSize / 2);
                state.ConsecutiveSuccesses = 0;
                System.GC.Collect();
                System.GC.WaitForPendingFinalizers();
                System.GC.Collect();
            }
        }
    }

    /// <summary>
    /// Returns <see langword="true"/> when <paramref name="input"/> is
    /// likely an unbatched single sample for <paramref name="nn"/> — its
    /// rank matches the architecture's expected unbatched input rank
    /// rather than that rank + 1. Used to short-circuit axis-0 chunking
    /// paths for inputs whose leading axis is sequence / features, not
    /// batch, even when that axis happens to exceed the chunk size.
    /// </summary>
    internal static bool IsLikelyUnbatched<T>(NeuralNetworkBase<T> nn, Tensor<T> input)
    {
        int expectedUnbatchedRank = nn.GetExpectedUnbatchedInputRankInternal();
        return expectedUnbatchedRank > 0 && input.Rank == expectedUnbatchedRank;
    }

    private static void TrainIndependentChunks<T>(
        NeuralNetworkBase<T> nn, Tensor<T> xTensor, Tensor<T> yTensor, int batchSize)
    {
        int n = xTensor.Shape[0];
        int nChunks = (n + batchSize - 1) / batchSize;
        for (int chunkIdx = 0; chunkIdx < nChunks; chunkIdx++)
        {
            int start = chunkIdx * batchSize;
            int end = System.Math.Min(start + batchSize, n);
            var xChunk = xTensor.Slice(axis: 0, start: start, end: end).Contiguous();
            var yChunk = yTensor.Slice(axis: 0, start: start, end: end).Contiguous();
            nn.Train(xChunk, yChunk);
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Feature 1: Adaptive OOM recovery
    // ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Alias for <see cref="PredictMaybeBatched{T,TInput,TOutput}"/> with
    /// adaptive OOM recovery — kept as a named entry point for callers
    /// that want to make the adaptive intent explicit. Adaptive recovery
    /// is now the default in <see cref="PredictMaybeBatched{T,TInput,TOutput}"/>
    /// so users no longer need to opt in. <paramref name="minBatchSize"/>
    /// caps how far the halve-on-OOM loop will reduce the chunk size
    /// before rethrowing — defaults to <see cref="MinAdaptiveBatchSize"/>.
    /// </summary>
    public static TOutput PredictAdaptive<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        TInput X,
        int initialBatchSize = DefaultBatchSize,
        int minBatchSize = MinAdaptiveBatchSize)
        => PredictMaybeBatched(model, X, initialBatchSize, disableAdaptiveRetry: false, minBatchSize: minBatchSize);

    /// <summary>
    /// Alias for <see cref="TrainMaybeBatched{T,TInput,TOutput}"/> with
    /// adaptive OOM recovery — adaptive is now the default; this
    /// overload preserves the explicit intent for callers that want it.
    /// <paramref name="minBatchSize"/> caps how far the halve-on-OOM loop
    /// will reduce the chunk size before rethrowing — defaults to
    /// <see cref="MinAdaptiveBatchSize"/>.
    /// </summary>
    public static void TrainAdaptive<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        TInput X,
        TOutput Y,
        int initialBatchSize = DefaultBatchSize,
        int minBatchSize = MinAdaptiveBatchSize,
        GradientAccumulationMode mode = GradientAccumulationMode.Independent)
        => TrainMaybeBatched(model, X, Y, initialBatchSize, mode, disableAdaptiveRetry: false, minBatchSize: minBatchSize);

    private static AdaptiveState GetOrCreateAdaptiveState(object modelKey, int initialBatchSize)
    {
        if (!_adaptiveStates.TryGetValue(modelKey, out var state))
        {
            state = new AdaptiveState { CurrentBatchSize = initialBatchSize };
            _adaptiveStates.Add(modelKey, state);
        }
        return state;
    }

    private static void MaybeGrowBatch(AdaptiveState state, int ceiling)
    {
        if (state.CurrentBatchSize >= ceiling) return;
        if (state.ConsecutiveSuccesses < SuccessesBeforeGrow) return;
        state.CurrentBatchSize = System.Math.Min(ceiling, state.CurrentBatchSize * 2);
        state.ConsecutiveSuccesses = 0;
    }

    // ─────────────────────────────────────────────────────────────────
    // Feature 2: Stream-aggregation
    // ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Walks <paramref name="X"/> in axis-0 chunks, runs
    /// <see cref="NeuralNetworkBase{T}.Predict"/> on each, and folds the
    /// per-chunk output into <paramref name="seed"/> via
    /// <paramref name="reducer"/>. The full concatenated prediction tensor
    /// is never materialised — eliminates the transient 2× peak
    /// <see cref="NeuralNetworkBase{T}.PredictInBatches"/> incurs when the
    /// caller only needs a scalar aggregate.
    /// </summary>
    public static TAccumulator PredictAndReduce<T, TInput, TOutput, TAccumulator>(
        IFullModel<T, TInput, TOutput> model,
        TInput X,
        TAccumulator seed,
        Func<TAccumulator, TOutput, int, int, TAccumulator> reducer,
        int batchSize = DefaultBatchSize)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        if (reducer is null) throw new ArgumentNullException(nameof(reducer));
        if (batchSize < 1) batchSize = 1;

        var nnOpt = model as NeuralNetworkBase<T>;
        var xTensorOpt = X as Tensor<T>;
        if (nnOpt is null || xTensorOpt is null || xTensorOpt.Rank < 1)
        {
            var single = model.Predict(X);
            int n = xTensorOpt is not null && xTensorOpt.Rank > 0 ? xTensorOpt.Shape[0] : 1;
            return reducer(seed, single, 0, n);
        }
        var nn = nnOpt;
        var xTensor = xTensorOpt;

        int totalN = xTensor.Shape[0];
        if (totalN <= batchSize)
        {
            var direct = nn.Predict(xTensor);
            if (direct is TOutput dt) return reducer(seed, dt, 0, totalN);
            return reducer(seed, model.Predict(X), 0, totalN);
        }

        TAccumulator acc = seed;
        int nChunks = (totalN + batchSize - 1) / batchSize;
        for (int chunkIdx = 0; chunkIdx < nChunks; chunkIdx++)
        {
            int start = chunkIdx * batchSize;
            int end = System.Math.Min(start + batchSize, totalN);
            var xChunk = xTensor.Slice(axis: 0, start: start, end: end).Contiguous();
            var chunkOut = nn.Predict(xChunk);
            if (chunkOut is TOutput typed)
            {
                acc = reducer(acc, typed, start, end);
            }
            else
            {
                // Per-chunk forward returned a runtime type incompatible
                // with TOutput. Discarding the accumulated reductions and
                // silently restarting from `seed` against a full-input
                // Predict would: (a) lose the chunked work already done,
                // (b) defeat the memory-bounding contract by running an
                // unchunked Predict, and (c) hide the type mismatch from
                // the caller. Fail fast with an actionable diagnostic.
                throw new InvalidOperationException(
                    $"PredictAndReduce: per-chunk Predict on chunk [{start}..{end}) " +
                    $"returned {chunkOut?.GetType().FullName ?? "null"}, which is not " +
                    $"assignable to TOutput={typeof(TOutput).FullName}. The reducer " +
                    $"contract requires every chunk output to be a {typeof(TOutput).Name}.");
            }
        }
        return acc;
    }

    // ─────────────────────────────────────────────────────────────────
    // Feature 4: Memory-budget API
    // ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Predicts <paramref name="X"/> at a chunk size sized to keep peak
    /// managed-heap delta under <paramref name="memoryBudgetBytes"/>.
    /// Probes per-sample cost via a B=1 forward, then computes the chunk
    /// that fits with <see cref="MemoryBudgetSafetyFactor"/> headroom.
    /// </summary>
    public static TOutput PredictWithMemoryBudget<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        TInput X,
        long memoryBudgetBytes)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        if (memoryBudgetBytes <= 0)
            throw new ArgumentOutOfRangeException(nameof(memoryBudgetBytes), "Memory budget must be positive.");

        if (model is NeuralNetworkBase<T> nn
            && X is Tensor<T> xTensor
            && xTensor.Rank >= 1
            && xTensor.Shape[0] > 1)
        {
            int chunk = EstimateChunkSize(nn, xTensor, memoryBudgetBytes);
            if (chunk >= xTensor.Shape[0])
            {
                var direct = nn.Predict(xTensor);
                return direct is TOutput dt ? dt : model.Predict(X);
            }
            var chunked = nn.PredictInBatches(xTensor, chunk);
            return chunked is TOutput typed ? typed : model.Predict(X);
        }
        return model.Predict(X);
    }

    /// <summary>
    /// Empirically estimates the chunk size that keeps a Predict call's
    /// peak managed-heap delta under <paramref name="memoryBudgetBytes"/>.
    /// Probes at <c>B = probeSmall</c> AND <c>B = probeLarge</c>, fits a
    /// linear slope <c>bytes(B) ≈ alpha + beta · B</c> via the two-point
    /// estimator, then solves for B such that <c>alpha + beta · B ≤
    /// budget × safety</c>.
    ///
    /// <para>
    /// A single B=1 probe systematically under-counts memory cost on
    /// quadratic-in-seq operators (attention scores) because the per-call
    /// fixed overhead (arena setup, layer caches, tape allocation) is a
    /// large fraction of B=1's measurement and masks the per-sample
    /// contribution. The two-point fit isolates the per-sample slope
    /// (beta) from the per-call fixed overhead (alpha), which is what
    /// actually determines whether a larger chunk fits.
    /// </para>
    /// </summary>
    public static int EstimateChunkSize<T>(
        NeuralNetworkBase<T> nn,
        Tensor<T> sampleInput,
        long memoryBudgetBytes)
    {
        if (nn is null) throw new ArgumentNullException(nameof(nn));
        if (sampleInput is null) throw new ArgumentNullException(nameof(sampleInput));
        if (sampleInput.Rank == 0) return 1;

        int axis0 = sampleInput.Shape[0];
        // Probe sizes — small enough to be feasible probes; large enough
        // that the per-sample slope dominates the fixed overhead in the
        // delta between them. If the input is small, fall back to a
        // single-point estimate at the input's own size.
        int probeSmall = System.Math.Min(8, axis0);
        int probeLarge = System.Math.Min(16, axis0);

        long bytesSmall = MeasureForwardAllocatedBytes(nn, sampleInput, probeSmall);
        long beta;       // per-sample slope (bytes per additional sample)
        long alpha;      // per-call fixed cost (intercept)
        if (probeLarge > probeSmall)
        {
            long bytesLarge = MeasureForwardAllocatedBytes(nn, sampleInput, probeLarge);
            long dB = probeLarge - probeSmall;
            beta = System.Math.Max(1L, (bytesLarge - bytesSmall) / dB);
            // Solve alpha from one point: alpha = bytesSmall - beta * probeSmall
            alpha = System.Math.Max(0L, bytesSmall - beta * probeSmall);
        }
        else
        {
            // Single-point fallback: assume zero fixed overhead and
            // attribute all measured bytes to the per-sample slope.
            beta = System.Math.Max(1L, bytesSmall / System.Math.Max(1, probeSmall));
            alpha = 0L;
        }

        long budgetWithMargin = (long)(memoryBudgetBytes * MemoryBudgetSafetyFactor);
        // Solve alpha + beta * chunk <= budget for chunk.
        long chunk = (budgetWithMargin - alpha) / beta;
        if (chunk < 1) return 1;
        if (chunk > axis0) return axis0;
        return (int)chunk;
    }

    /// <summary>
    /// Probes the actual bytes-allocated cost of a single Predict at
    /// <paramref name="batchSize"/> samples. Uses
    /// <see cref="System.GC.GetAllocatedBytesForCurrentThread"/> which
    /// captures every allocation regardless of GC timing — strictly
    /// better than the heap-delta read used previously.
    /// </summary>
    private static long MeasureForwardAllocatedBytes<T>(
        NeuralNetworkBase<T> nn, Tensor<T> sampleInput, int batchSize)
    {
        var probeInput = sampleInput.Shape[0] >= batchSize
            ? sampleInput.Slice(axis: 0, start: 0, end: batchSize).Contiguous()
            : sampleInput;

        // Warm-up: run once so JIT and lazy-init costs don't pollute the
        // probe. The first Predict on a lazy-init Transformer allocates
        // weights — only the SECOND call reflects steady-state forward
        // cost.
        _ = nn.Predict(probeInput);

        // GetAllocatedBytesForCurrentThread is net5+; for net471 fall
        // back to GetTotalMemory (less precise but always available).
#if NET5_0_OR_GREATER
        long before = System.GC.GetAllocatedBytesForCurrentThread();
        var probeOutput = nn.Predict(probeInput);
        long after = System.GC.GetAllocatedBytesForCurrentThread();
#else
        long before = System.GC.GetTotalMemory(forceFullCollection: true);
        var probeOutput = nn.Predict(probeInput);
        long after = System.GC.GetTotalMemory(forceFullCollection: false);
#endif
        if (probeOutput is System.IDisposable disposable) disposable.Dispose();
        return System.Math.Max(0L, after - before);
    }
}
