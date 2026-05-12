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
/// Exposes five capabilities beyond the basic per-call chunking that
/// closed the #1296 OOM family. PyTorch / TensorFlow ship none of these
/// out of the box (1, 3, 4 are user-rolled patterns; 2 is unique; 5
/// supersedes the value-stability hazard that kept compile-on-Predict
/// off-by-default before this PR):
/// </para>
/// <list type="number">
///   <item><b>Adaptive OOM recovery</b> (<see cref="PredictAdaptive{T,TInput,TOutput}"/>,
///     <see cref="TrainAdaptive{T,TInput,TOutput}"/>) — catches
///     <see cref="OutOfMemoryException"/>, halves the chunk size, retries.
///     Auto-grows back up on successive successes via a per-model ratchet
///     held in a weak-reference table.</item>
///   <item><b>Stream-aggregation</b> (<see cref="PredictAndReduce{T,TInput,TOutput,TAccumulator}"/>) —
///     per-chunk reducer callback that never materialises the full
///     predictions tensor. The optimizer's R² / SS<sub>res</sub> path
///     uses it to drop the concat 2× peak the plain
///     <see cref="NeuralNetworkBase{T}.PredictInBatches"/> incurs.</item>
///   <item><b>True gradient accumulation</b> (<see cref="GradientAccumulationMode.Accumulate"/>) —
///     delegates to <see cref="NeuralNetworkBase{T}.TrainWithGradientAccumulation"/>
///     so the optimizer fires exactly once with averaged chunk gradients,
///     preserving full-batch direction AND keeping Adam's m/v cadence at
///     one update per logical batch.</item>
///   <item><b>Memory-budget API</b> (<see cref="PredictWithMemoryBudget{T,TInput,TOutput}"/>,
///     <see cref="EstimateChunkSize{T}"/>) — caller specifies a budget
///     in bytes; the helper probes per-sample cost with a B=1 forward
///     and picks the chunk size that fits with safety margin.</item>
///   <item><b>Value-stable compile reuse</b> — <see cref="NeuralNetworkBase{T}.PredictInBatches"/>
///     now routes per-chunk forwards through <see cref="NeuralNetworkBase{T}.PredictCompiled"/>
///     when <see cref="AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions"/>
///     enables compilation. The companion <see cref="CompiledModelHost{T}.Predict"/>
///     fix in this PR copies the current call's data into the cached plan's
///     captured input buffer via <see cref="AiDotNet.Tensors.Engines.Compilation.ICompiledPlan{T}.SetInputs"/>
///     on every replay, eliminating the same-shape-different-values stale-
///     data hazard that previously kept compile-by-default off the
///     <c>Predict</c> path.</item>
/// </list>
/// </summary>
public static class NeuralBatchHelper
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
        bool disableAdaptiveRetry = false)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        if (batchSize < 1) batchSize = 1;

        if (!(model is NeuralNetworkBase<T> nn
            && X is Tensor<T> xTensor
            && xTensor.Rank >= 1))
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
                if (state.CurrentBatchSize <= MinAdaptiveBatchSize) throw;
                state.CurrentBatchSize = System.Math.Max(MinAdaptiveBatchSize, state.CurrentBatchSize / 2);
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
        bool disableAdaptiveRetry = false)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        if (batchSize < 1) batchSize = 1;

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
                if (state.CurrentBatchSize <= MinAdaptiveBatchSize) throw;
                state.CurrentBatchSize = System.Math.Max(MinAdaptiveBatchSize, state.CurrentBatchSize / 2);
                state.ConsecutiveSuccesses = 0;
                System.GC.Collect();
                System.GC.WaitForPendingFinalizers();
                System.GC.Collect();
            }
        }
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
    /// so users no longer need to opt in.
    /// </summary>
    public static TOutput PredictAdaptive<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        TInput X,
        int initialBatchSize = DefaultBatchSize,
        int minBatchSize = MinAdaptiveBatchSize)
        => PredictMaybeBatched(model, X, initialBatchSize, disableAdaptiveRetry: false);

    /// <summary>
    /// Alias for <see cref="TrainMaybeBatched{T,TInput,TOutput}"/> with
    /// adaptive OOM recovery — adaptive is now the default; this
    /// overload preserves the explicit intent for callers that want it.
    /// </summary>
    public static void TrainAdaptive<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        TInput X,
        TOutput Y,
        int initialBatchSize = DefaultBatchSize,
        int minBatchSize = MinAdaptiveBatchSize,
        GradientAccumulationMode mode = GradientAccumulationMode.Independent)
        => TrainMaybeBatched(model, X, Y, initialBatchSize, mode, disableAdaptiveRetry: false);

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
                return reducer(seed, model.Predict(X), 0, totalN);
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
    /// Probes with a single-sample forward, scales linearly against the
    /// budget with <see cref="MemoryBudgetSafetyFactor"/> headroom.
    /// </summary>
    public static int EstimateChunkSize<T>(
        NeuralNetworkBase<T> nn,
        Tensor<T> sampleInput,
        long memoryBudgetBytes)
    {
        if (nn is null) throw new ArgumentNullException(nameof(nn));
        if (sampleInput is null) throw new ArgumentNullException(nameof(sampleInput));
        if (sampleInput.Rank == 0) return 1;

        var oneSample = sampleInput.Shape[0] == 1
            ? sampleInput
            : sampleInput.Slice(axis: 0, start: 0, end: 1).Contiguous();

        long before = System.GC.GetTotalMemory(forceFullCollection: true);
        var probeOutput = nn.Predict(oneSample);
        long after = System.GC.GetTotalMemory(forceFullCollection: false);
        long perSampleBytes = System.Math.Max(1L, after - before);

        if (probeOutput is System.IDisposable disposable) disposable.Dispose();

        long budgetWithMargin = (long)(memoryBudgetBytes * MemoryBudgetSafetyFactor);
        long chunk = budgetWithMargin / perSampleBytes;
        if (chunk < 1) return 1;
        if (chunk > sampleInput.Shape[0]) return sampleInput.Shape[0];
        return (int)chunk;
    }
}
