using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.DistributedTraining;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Wraps any <see cref="IStreamingDataLoader{T, TInput, TOutput}"/> to partition each global batch deterministically
/// across distributed ranks, supporting both homogeneous and heterogeneous per-rank micro-batch sizes as well as
/// exact sample-offset fast-forwarding across world-size changes on checkpoint resume.
/// </summary>
/// <typeparam name="T">The numeric element type used by the communication backend.</typeparam>
/// <typeparam name="TInput">The input sample type.</typeparam>
/// <typeparam name="TOutput">The target/label sample type.</typeparam>
/// <remarks>
/// <para>
/// At the start of enumeration, ranks exchange their configured local <see cref="BatchSize"/> values
/// (<c>B_0, B_1, ..., B_{W-1}</c>) via <see cref="ICommunicationBackend{T}.AllGather"/> so every rank knows
/// <see cref="GlobalBatchSize"/> (<c>sum_r B_r</c>). For every global step, the underlying sample stream is
/// partitioned contiguously across ranks <c>0..W-1</c>, ensuring the union of per-rank micro-batches equals a
/// single-process run configured with <c>BatchSize = GlobalBatchSize</c> and the same shuffle seed.
/// </para>
/// <para><b>For Beginners:</b> If GPU 0 (12 GB) uses <c>BatchSize = 2</c> and GPU 1 (24 GB) uses <c>BatchSize = 6</c>,
/// <see cref="DistributedStreamingDataLoader{T, TInput, TOutput}"/> automatically forms a global batch of <c>8</c>
/// non-overlapping samples per step—feeding samples <c>0..1</c> to GPU 0 and samples <c>2..7</c> to GPU 1—without
/// duplicating data or hanging on incomplete tail batches.</para>
/// </remarks>
public sealed class DistributedStreamingDataLoader<T, TInput, TOutput> : IStreamingDataLoader<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IStreamingDataLoader<T, TInput, TOutput> _innerLoader;
    private readonly ICommunicationBackend<T> _backend;
    private int _localBatchSize;
    private int[] _rankBatchSizes;
    private int _globalBatchSize;
    private bool _batchPlanSynchronized;
    private long _pendingSkipSamples;
    private long _lastBatchGlobalSampleCount;
    private int _currentBatchIndex;
    private long _currentGlobalSampleIndex;

    /// <summary>
    /// Initializes a new <see cref="DistributedStreamingDataLoader{T, TInput, TOutput}"/> around an existing streaming loader.
    /// </summary>
    /// <param name="innerLoader">The underlying streaming data loader.</param>
    /// <param name="backend">The communication backend identifying this process's rank and world size.</param>
    /// <param name="localBatchSize">
    /// Optional per-rank micro-batch size override; defaults to <paramref name="innerLoader"/>'s current <see cref="IStreamingDataLoader{T, TInput, TOutput}.BatchSize"/>.
    /// </param>
    public DistributedStreamingDataLoader(
        IStreamingDataLoader<T, TInput, TOutput> innerLoader,
        ICommunicationBackend<T> backend,
        int? localBatchSize = null)
    {
        _innerLoader = innerLoader ?? throw new ArgumentNullException(nameof(innerLoader));
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));

        int resolvedLocalBatch = localBatchSize ?? innerLoader.BatchSize;
        if (resolvedLocalBatch <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(localBatchSize), "Per-rank batch size must be positive.");
        }

        _localBatchSize = resolvedLocalBatch;
        _rankBatchSizes = new[] { resolvedLocalBatch };
        _globalBatchSize = resolvedLocalBatch * Math.Max(1, backend.WorldSize);
        _batchPlanSynchronized = backend.WorldSize <= 1;
    }

    /// <summary>
    /// Gets the underlying unwrapped streaming data loader.
    /// </summary>
    public IStreamingDataLoader<T, TInput, TOutput> InnerLoader => _innerLoader;

    /// <summary>
    /// Gets this process's rank in <c>[0, WorldSize - 1]</c>.
    /// </summary>
    public int Rank => _backend.Rank;

    /// <summary>
    /// Gets the number of participating distributed processes.
    /// </summary>
    public int WorldSize => _backend.WorldSize;

    /// <summary>
    /// Gets the synchronized global batch size across all ranks (<c>sum_r RankBatchSizes[r]</c>).
    /// </summary>
    public int GlobalBatchSize
    {
        get
        {
            EnsureBatchPlanSynchronized();
            return _globalBatchSize;
        }
    }

    /// <summary>
    /// Gets the per-rank micro-batch sizes (<c>[B_0, ..., B_{W-1}]</c>) synchronized across the world.
    /// </summary>
    public IReadOnlyList<int> RankBatchSizes
    {
        get
        {
            EnsureBatchPlanSynchronized();
            return _rankBatchSizes;
        }
    }

    /// <summary>
    /// Gets the total number of global samples consumed across all ranks in the most recently yielded step.
    /// </summary>
    public long LastBatchGlobalSampleCount => Interlocked.Read(ref _lastBatchGlobalSampleCount);

    /// <inheritdoc/>
    public string Name => _innerLoader.Name;

    /// <inheritdoc/>
    public string Description => _innerLoader.Description;

    /// <inheritdoc/>
    public bool IsLoaded => _innerLoader.IsLoaded;

    /// <inheritdoc/>
    public int SampleCount => _innerLoader.SampleCount;

    /// <inheritdoc/>
    public int TotalCount => _innerLoader.TotalCount;

    /// <inheritdoc/>
    public int CurrentIndex => (int)Math.Min(int.MaxValue, Interlocked.Read(ref _currentGlobalSampleIndex));

    /// <inheritdoc/>
    public int CurrentBatchIndex => Volatile.Read(ref _currentBatchIndex);

    /// <inheritdoc/>
    public double Progress => TotalCount > 0 ? Math.Min(1.0, (double)Interlocked.Read(ref _currentGlobalSampleIndex) / TotalCount) : 0.0;

    /// <inheritdoc/>
    public int BatchCount
    {
        get
        {
            if (TotalCount <= 0)
            {
                return 0;
            }

            int globalBs = _batchPlanSynchronized ? _globalBatchSize : (_localBatchSize * Math.Max(1, _backend.WorldSize));
            return (TotalCount + globalBs - 1) / Math.Max(1, globalBs);
        }
    }

    /// <inheritdoc/>
    public int BatchSize
    {
        get => _localBatchSize;
        set
        {
            if (value <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(value), "Batch size must be positive.");
            }

            if (_localBatchSize != value)
            {
                _localBatchSize = value;
                _batchPlanSynchronized = _backend.WorldSize <= 1;
                if (_backend.WorldSize <= 1)
                {
                    _rankBatchSizes = new[] { value };
                    _globalBatchSize = value;
                }
            }
        }
    }

    /// <inheritdoc/>
    public int PrefetchCount => _innerLoader.PrefetchCount;

    /// <inheritdoc/>
    public int NumWorkers => _innerLoader.NumWorkers;

    /// <summary>
    /// Schedules an exact global sample count to skip at the start of the next <see cref="GetBatches"/> or
    /// <see cref="GetBatchesAsync"/> call (used when resuming mid-epoch from a checkpoint across arbitrary world sizes).
    /// </summary>
    /// <param name="samplesToSkip">Number of global samples already trained in the current epoch.</param>
    public void SkipSamplesOnNextEnumeration(long samplesToSkip)
    {
        if (samplesToSkip < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(samplesToSkip), "Samples to skip cannot be negative.");
        }

        Interlocked.Exchange(ref _pendingSkipSamples, samplesToSkip);
    }

    /// <inheritdoc/>
    public Task LoadAsync(CancellationToken cancellationToken = default) => _innerLoader.LoadAsync(cancellationToken);

    /// <inheritdoc/>
    public void Unload() => _innerLoader.Unload();

    /// <inheritdoc/>
    public void Reset()
    {
        Volatile.Write(ref _currentBatchIndex, 0);
        Interlocked.Exchange(ref _currentGlobalSampleIndex, 0L);
        Interlocked.Exchange(ref _lastBatchGlobalSampleCount, 0L);
        _innerLoader.Reset();
    }

    /// <inheritdoc/>
    public IEnumerable<(TInput[] Inputs, TOutput[] Outputs)> GetBatches(
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null)
    {
        EnsureBatchPlanSynchronized();
        int? resolvedSeed = SynchronizeEpochSeed(shuffle, seed);
        long skipSamples = Interlocked.Exchange(ref _pendingSkipSamples, 0L);
        Volatile.Write(ref _currentBatchIndex, 0);
        Interlocked.Exchange(ref _currentGlobalSampleIndex, skipSamples);

        if (CanUseBaseDirectSliceSync(_innerLoader, out var baseLoader))
        {
            foreach (var item in baseLoader.GetDistributedSliceBatchesCore(
                _backend.Rank,
                _rankBatchSizes,
                _globalBatchSize,
                skipSamples,
                shuffle,
                dropLast,
                resolvedSeed))
            {
                RecordStepAdvance(item.GlobalBatchSamples);
                yield return (item.Inputs, item.Outputs);
            }

            yield break;
        }

        foreach (var item in EnumerateFallbackSync(skipSamples, shuffle, dropLast, resolvedSeed))
        {
            RecordStepAdvance(item.GlobalBatchSamples);
            yield return (item.Inputs, item.Outputs);
        }
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<(TInput[] Inputs, TOutput[] Outputs)> GetBatchesAsync(
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        EnsureBatchPlanSynchronized();
        int? resolvedSeed = SynchronizeEpochSeed(shuffle, seed);
        long skipSamples = Interlocked.Exchange(ref _pendingSkipSamples, 0L);
        Volatile.Write(ref _currentBatchIndex, 0);
        Interlocked.Exchange(ref _currentGlobalSampleIndex, skipSamples);

        if (CanUseBaseDirectSliceAsync(_innerLoader, out var baseLoader))
        {
            await foreach (var item in baseLoader.GetDistributedSliceBatchesAsyncCore(
                _backend.Rank,
                _rankBatchSizes,
                _globalBatchSize,
                skipSamples,
                shuffle,
                dropLast,
                resolvedSeed,
                cancellationToken).ConfigureAwait(false))
            {
                RecordStepAdvance(item.GlobalBatchSamples);
                yield return (item.Inputs, item.Outputs);
            }

            yield break;
        }

        await foreach (var item in EnumerateFallbackAsync(skipSamples, shuffle, dropLast, resolvedSeed, cancellationToken).ConfigureAwait(false))
        {
            RecordStepAdvance(item.GlobalBatchSamples);
            yield return (item.Inputs, item.Outputs);
        }
    }

    /// <summary>
    /// Computes the contiguous slice <c>[rankOffset .. rankOffset + rankCount)</c> assigned to <paramref name="rank"/>
    /// for a global step containing <paramref name="globalCount"/> samples.
    /// </summary>
    internal static void ComputeRankSlice(
        int globalCount,
        int[] rankBatchSizes,
        int globalBatchSize,
        int rank,
        out int rankOffset,
        out int rankCount)
    {
        if (globalCount <= 0)
        {
            rankOffset = 0;
            rankCount = 0;
            return;
        }

        if (globalCount == globalBatchSize)
        {
            int offset = 0;
            for (int r = 0; r < rank; r++)
            {
                offset += rankBatchSizes[r];
            }

            rankOffset = offset;
            rankCount = rankBatchSizes[rank];
            return;
        }

        // Tail step (0 < globalCount < globalBatchSize): allocate proportionally without exceeding any rank's BatchSize.
        int worldSize = rankBatchSizes.Length;
        var counts = new int[worldSize];
        int assigned = 0;
        for (int r = 0; r < worldSize; r++)
        {
            int c = Math.Min(rankBatchSizes[r], (int)((long)globalCount * rankBatchSizes[r] / globalBatchSize));
            counts[r] = c;
            assigned += c;
        }

        while (assigned < globalCount)
        {
            bool progressed = false;
            for (int r = 0; r < worldSize && assigned < globalCount; r++)
            {
                if (counts[r] < rankBatchSizes[r])
                {
                    counts[r]++;
                    assigned++;
                    progressed = true;
                }
            }

            if (!progressed)
            {
                break;
            }
        }

        int tailOffset = 0;
        for (int r = 0; r < rank; r++)
        {
            tailOffset += counts[r];
        }

        rankOffset = tailOffset;
        rankCount = counts[rank];
    }

    [AiDotNet.Attributes.ScalarKernel("One-time WorldSize batch-size metadata unpack")]
    private void EnsureBatchPlanSynchronized()
    {
        if (_batchPlanSynchronized)
        {
            return;
        }

        if (!_backend.IsInitialized)
        {
            _backend.Initialize();
        }

        if (_backend.WorldSize <= 1)
        {
            _rankBatchSizes = new[] { _localBatchSize };
            _globalBatchSize = _localBatchSize;
            _batchPlanSynchronized = true;
            return;
        }

        var localVec = new Vector<T>(new[] { NumOps.FromDouble(_localBatchSize) });
        var gathered = _backend.AllGather(localVec);
        var span = gathered.AsSpan();
        var sizes = new int[_backend.WorldSize];
        long total = 0;
        for (int r = 0; r < _backend.WorldSize; r++)
        {
            int bs = (int)Math.Round(NumOps.ToDouble(span[r]));
            if (bs <= 0)
            {
                throw new InvalidOperationException($"Rank {r} reported non-positive batch size {bs}.");
            }

            sizes[r] = bs;
            total += bs;
        }

        _rankBatchSizes = sizes;
        _globalBatchSize = checked((int)total);
        _batchPlanSynchronized = true;
    }

    private int? SynchronizeEpochSeed(bool shuffle, int? seed)
    {
        if (!shuffle || seed.HasValue || _backend.WorldSize <= 1)
        {
            return seed;
        }

        int generated = _backend.Rank == 0
            ? RandomHelper.CreateSecureRandom().Next(1, int.MaxValue)
            : 0;
        var seedVec = _backend.Broadcast(new Vector<T>(new[] { NumOps.FromDouble(generated) }), root: 0);
        return (int)Math.Round(NumOps.ToDouble(seedVec[0]));
    }

    private void RecordStepAdvance(int globalSamples)
    {
        Interlocked.Exchange(ref _lastBatchGlobalSampleCount, globalSamples);
        Interlocked.Add(ref _currentGlobalSampleIndex, globalSamples);
        Interlocked.Increment(ref _currentBatchIndex);
    }

    private static bool CanUseBaseDirectSliceSync(
        IStreamingDataLoader<T, TInput, TOutput> loader,
        out StreamingDataLoaderBase<T, TInput, TOutput> baseLoader)
    {
        if (loader is StreamingDataLoaderBase<T, TInput, TOutput> candidate)
        {
            var method = candidate.GetType().GetMethod(
                nameof(IStreamingDataLoader<T, TInput, TOutput>.GetBatches),
                BindingFlags.Instance | BindingFlags.Public,
                binder: null,
                types: new[] { typeof(bool), typeof(bool), typeof(int?) },
                modifiers: null);
            if (method?.DeclaringType == typeof(StreamingDataLoaderBase<T, TInput, TOutput>))
            {
                baseLoader = candidate;
                return true;
            }
        }

        baseLoader = null!;
        return false;
    }

    private static bool CanUseBaseDirectSliceAsync(
        IStreamingDataLoader<T, TInput, TOutput> loader,
        out StreamingDataLoaderBase<T, TInput, TOutput> baseLoader)
    {
        if (loader is StreamingDataLoaderBase<T, TInput, TOutput> candidate)
        {
            var method = candidate.GetType().GetMethod(
                nameof(IStreamingDataLoader<T, TInput, TOutput>.GetBatchesAsync),
                BindingFlags.Instance | BindingFlags.Public,
                binder: null,
                types: new[] { typeof(bool), typeof(bool), typeof(int?), typeof(CancellationToken) },
                modifiers: null);
            if (method?.DeclaringType == typeof(StreamingDataLoaderBase<T, TInput, TOutput>))
            {
                baseLoader = candidate;
                return true;
            }
        }

        baseLoader = null!;
        return false;
    }

    private IEnumerable<(TInput[] Inputs, TOutput[] Outputs, int GlobalBatchSamples)> EnumerateFallbackSync(
        long skipSamples,
        bool shuffle,
        bool dropLast,
        int? seed)
    {
        int savedBatchSize = _innerLoader.BatchSize;
        _innerLoader.BatchSize = _globalBatchSize;
        try
        {
            var inputBuf = new Queue<TInput>(_globalBatchSize * 2);
            var outputBuf = new Queue<TOutput>(_globalBatchSize * 2);
            long remainingToSkip = skipSamples;

            foreach (var (inputs, outputs) in _innerLoader.GetBatches(shuffle, dropLast: false, seed))
            {
                int count = Math.Min(inputs.Length, outputs.Length);
                int idx = 0;
                if (remainingToSkip > 0)
                {
                    int skipNow = (int)Math.Min(remainingToSkip, count);
                    remainingToSkip -= skipNow;
                    idx = skipNow;
                }

                for (; idx < count; idx++)
                {
                    inputBuf.Enqueue(inputs[idx]);
                    outputBuf.Enqueue(outputs[idx]);
                }

                while (inputBuf.Count >= _globalBatchSize)
                {
                    yield return SliceFromQueues(inputBuf, outputBuf, _globalBatchSize);
                }
            }

            if (inputBuf.Count > 0 && !dropLast)
            {
                yield return SliceFromQueues(inputBuf, outputBuf, inputBuf.Count);
            }
        }
        finally
        {
            _innerLoader.BatchSize = savedBatchSize;
        }
    }

    private async IAsyncEnumerable<(TInput[] Inputs, TOutput[] Outputs, int GlobalBatchSamples)> EnumerateFallbackAsync(
        long skipSamples,
        bool shuffle,
        bool dropLast,
        int? seed,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        int savedBatchSize = _innerLoader.BatchSize;
        _innerLoader.BatchSize = _globalBatchSize;
        try
        {
            var inputBuf = new Queue<TInput>(_globalBatchSize * 2);
            var outputBuf = new Queue<TOutput>(_globalBatchSize * 2);
            long remainingToSkip = skipSamples;

            await foreach (var (inputs, outputs) in _innerLoader.GetBatchesAsync(shuffle, dropLast: false, seed, cancellationToken).ConfigureAwait(false))
            {
                int count = Math.Min(inputs.Length, outputs.Length);
                int idx = 0;
                if (remainingToSkip > 0)
                {
                    int skipNow = (int)Math.Min(remainingToSkip, count);
                    remainingToSkip -= skipNow;
                    idx = skipNow;
                }

                for (; idx < count; idx++)
                {
                    inputBuf.Enqueue(inputs[idx]);
                    outputBuf.Enqueue(outputs[idx]);
                }

                while (inputBuf.Count >= _globalBatchSize)
                {
                    yield return SliceFromQueues(inputBuf, outputBuf, _globalBatchSize);
                }
            }

            if (inputBuf.Count > 0 && !dropLast)
            {
                yield return SliceFromQueues(inputBuf, outputBuf, inputBuf.Count);
            }
        }
        finally
        {
            _innerLoader.BatchSize = savedBatchSize;
        }
    }

    private (TInput[] Inputs, TOutput[] Outputs, int GlobalBatchSamples) SliceFromQueues(
        Queue<TInput> inputBuf,
        Queue<TOutput> outputBuf,
        int globalCount)
    {
        ComputeRankSlice(
            globalCount,
            _rankBatchSizes,
            _globalBatchSize,
            _backend.Rank,
            out int rankOffset,
            out int rankCount);

        var sliceIn = rankCount > 0 ? new TInput[rankCount] : Array.Empty<TInput>();
        var sliceOut = rankCount > 0 ? new TOutput[rankCount] : Array.Empty<TOutput>();
        int sliceEnd = rankOffset + rankCount;

        for (int i = 0; i < globalCount; i++)
        {
            var itemIn = inputBuf.Dequeue();
            var itemOut = outputBuf.Dequeue();
            if (i >= rankOffset && i < sliceEnd)
            {
                int localIdx = i - rankOffset;
                sliceIn[localIdx] = itemIn;
                sliceOut[localIdx] = itemOut;
            }
        }

        return (sliceIn, sliceOut, globalCount);
    }
}
