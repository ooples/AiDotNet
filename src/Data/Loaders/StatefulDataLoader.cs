using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Wraps any <see cref="InMemoryDataLoader{T, TInput, TOutput}"/> with checkpoint/resume support.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Inspired by PyTorch's StatefulDataLoader (torchdata 2025), this wrapper adds
/// mid-epoch checkpointing to any in-memory data loader. The state can be serialized
/// and restored for fault-tolerant training on large datasets.
/// </para>
/// <para><b>For Beginners:</b> Wrap your data loader with this class to enable
/// saving and restoring the exact position during training:
/// <code>
/// var loader = DataLoaders.FromTensors(features, labels);
/// var stateful = new StatefulDataLoader&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;(loader);
///
/// // Train for a while...
/// var checkpoint = stateful.GetState();
///
/// // Later, after crash or restart:
/// stateful.LoadState(checkpoint);
/// // Continues exactly where it left off
/// </code>
/// </para>
/// </remarks>
public class StatefulDataLoader<T, TInput, TOutput> :
    DataLoaderBase<T>,
    IStatefulDataLoader<T>,
    IInputOutputDataLoader<T, TInput, TOutput>
{
    private readonly InputOutputDataLoaderBase<T, TInput, TOutput> _inner;
    private int _epoch;
    private int? _lastShuffleSeed;

    /// <summary>
    /// Creates a stateful wrapper around an existing data loader.
    /// </summary>
    /// <param name="inner">The data loader to wrap.</param>
    public StatefulDataLoader(InputOutputDataLoaderBase<T, TInput, TOutput> inner)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
    }

    /// <inheritdoc/>
    public override string Name => $"Stateful({_inner.Name})";

    /// <inheritdoc/>
    public override string Description => $"Stateful wrapper for {_inner.Description}";

    /// <inheritdoc/>
    public override int TotalCount => _inner.TotalCount;

    /// <inheritdoc/>
    public TInput Features => _inner.Features;

    /// <inheritdoc/>
    public TOutput Labels => _inner.Labels;

    /// <inheritdoc/>
    public int FeatureCount => _inner.FeatureCount;

    /// <inheritdoc/>
    public int OutputDimension => _inner.OutputDimension;

    /// <inheritdoc/>
    public bool HasNext => _inner.HasNext;

    /// <inheritdoc/>
    public bool IsShuffled => _inner.IsShuffled;

    /// <summary>
    /// Gets the current epoch number.
    /// </summary>
    public int Epoch => _epoch;

    /// <inheritdoc/>
    public DataLoaderCheckpoint GetState()
    {
        return new DataLoaderCheckpoint
        {
            CurrentIndex = _inner.CurrentIndex,
            CurrentBatchIndex = _inner.CurrentBatchIndex,
            Epoch = _epoch,
            RandomSeed = _lastShuffleSeed,
            TotalCount = TotalCount,
            BatchSize = _inner.BatchSize
        };
    }

    /// <inheritdoc/>
    public void LoadState(DataLoaderCheckpoint state)
    {
        if (state is null)
        {
            throw new ArgumentNullException(nameof(state));
        }

        if (state.TotalCount != TotalCount)
        {
            throw new InvalidOperationException(
                $"Checkpoint total count ({state.TotalCount}) does not match current loader ({TotalCount}).");
        }

        _epoch = state.Epoch;
        _lastShuffleSeed = state.RandomSeed;

        // Restore batch size
        _inner.BatchSize = state.BatchSize;

        // Reset and reload to the saved position
        _inner.Reset();

        // Restore the shuffle order deterministically from the saved seed
        if (state.RandomSeed.HasValue)
        {
            _inner.Shuffle(state.RandomSeed.Value);
        }

        // Advance to the saved position by consuming batches
        int batchesToSkip = state.CurrentBatchIndex;
        for (int i = 0; i < batchesToSkip && _inner.HasNext; i++)
        {
            _inner.GetNextBatch();
        }

        // Sync wrapper progress with inner loader
        CurrentIndex = _inner.CurrentIndex;
        CurrentBatchIndex = _inner.CurrentBatchIndex;
    }

    /// <inheritdoc/>
    public (TInput Features, TOutput Labels) GetNextBatch()
    {
        var result = _inner.GetNextBatch();
        CurrentIndex = _inner.CurrentIndex;
        CurrentBatchIndex = _inner.CurrentBatchIndex;
        return result;
    }

    /// <inheritdoc/>
    public bool TryGetNextBatch(out (TInput Features, TOutput Labels) batch)
    {
        bool hasNext = _inner.TryGetNextBatch(out batch);
        if (hasNext)
        {
            CurrentIndex = _inner.CurrentIndex;
            CurrentBatchIndex = _inner.CurrentBatchIndex;
        }
        return hasNext;
    }

    /// <inheritdoc/>
    public void Shuffle(int? seed = null)
    {
        // Always store a seed so checkpoints can reproduce the shuffle.
        // If no seed is provided, generate one deterministically.
        if (!seed.HasValue)
        {
            seed = Environment.TickCount;
        }

        _lastShuffleSeed = seed;
        _inner.Shuffle(seed);
    }

    /// <inheritdoc/>
    public void Unshuffle()
    {
        _lastShuffleSeed = null;
        _inner.Unshuffle();
    }

    /// <inheritdoc/>
    public (IInputOutputDataLoader<T, TInput, TOutput> Train,
        IInputOutputDataLoader<T, TInput, TOutput> Validation,
        IInputOutputDataLoader<T, TInput, TOutput> Test) Split(
        double trainRatio = 0.7,
        double validationRatio = 0.15,
        int? seed = null)
    {
        return _inner.Split(trainRatio, validationRatio, seed);
    }

    /// <inheritdoc/>
    public IEnumerable<(TInput Features, TOutput Labels)> GetBatches(
        int? batchSize = null,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null)
    {
        if (shuffle)
        {
            if (!seed.HasValue)
            {
                seed = Environment.TickCount;
            }

            _lastShuffleSeed = seed;
        }

        foreach (var batch in _inner.GetBatches(batchSize, shuffle, dropLast, seed))
        {
            CurrentIndex = _inner.CurrentIndex;
            CurrentBatchIndex = _inner.CurrentBatchIndex;
            yield return batch;
        }
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<(TInput Features, TOutput Labels)> GetBatchesAsync(
        int? batchSize = null,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null,
        int prefetchCount = 2,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (shuffle)
        {
            if (!seed.HasValue)
            {
                seed = Environment.TickCount;
            }

            _lastShuffleSeed = seed;
        }

        await foreach (var batch in _inner.GetBatchesAsync(
            batchSize, shuffle, dropLast, seed, prefetchCount, cancellationToken))
        {
            CurrentIndex = _inner.CurrentIndex;
            CurrentBatchIndex = _inner.CurrentBatchIndex;
            yield return batch;
        }
    }

    /// <summary>
    /// Called when starting a new epoch. Increments the epoch counter.
    /// </summary>
    public void OnEpochStart()
    {
        _epoch++;
        _inner.Reset();
    }

    /// <inheritdoc/>
    protected override Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        return _inner.LoadAsync(cancellationToken);
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        _inner.Unload();
        _epoch = 0;
        _lastShuffleSeed = null;
    }
}
