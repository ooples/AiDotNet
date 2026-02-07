using AiDotNet.Data.Loaders;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Formats;

/// <summary>
/// A typed data loader facade that wraps <see cref="ShardedStreamingDataset"/> and implements
/// <see cref="StreamingDataLoaderBase{T, TInput, TOutput}"/> for IDataLoader compliance.
/// </summary>
/// <remarks>
/// <para>
/// This loader eagerly reads all records from the underlying sharded binary files during loading,
/// caches them in memory, and provides index-based access via a user-supplied parser delegate
/// that converts raw <c>byte[]</c> records into typed tensor pairs.
/// </para>
/// <para><b>For Beginners:</b> Use this when you want to load sharded binary datasets and
/// feed them into <c>AiModelBuilder.ConfigureDataLoader()</c>. You provide a parser function
/// that knows how to decode each binary record into tensors for training.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for tensor values.</typeparam>
public class ShardedStreamingDataLoader<T> : StreamingDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly string[] _shardPaths;
    private readonly Func<byte[], (Tensor<T>, Tensor<T>)> _recordParser;
    private readonly ShardedStreamingDatasetOptions? _options;
    private List<byte[]> _cachedRecords = new();
    private ShardedStreamingDataset? _shardedDataset;

    /// <summary>
    /// Creates a new ShardedStreamingDataLoader.
    /// </summary>
    /// <param name="shardPaths">Paths to the shard files.</param>
    /// <param name="recordParser">
    /// A function that converts a raw byte record into a
    /// (features, labels) tensor pair.
    /// </param>
    /// <param name="batchSize">Number of samples per batch. Default is 32.</param>
    /// <param name="options">Optional sharded streaming configuration.</param>
    public ShardedStreamingDataLoader(
        string[] shardPaths,
        Func<byte[], (Tensor<T>, Tensor<T>)> recordParser,
        int batchSize = 32,
        ShardedStreamingDatasetOptions? options = null)
        : base(batchSize)
    {
        if (shardPaths is null || shardPaths.Length == 0)
            throw new ArgumentException("At least one shard path is required.", nameof(shardPaths));
        if (recordParser is null)
            throw new ArgumentNullException(nameof(recordParser));

        _shardPaths = shardPaths;
        _recordParser = recordParser;
        _options = options;
    }

    /// <inheritdoc/>
    public override string Name => "ShardedStreaming";

    /// <inheritdoc/>
    public override int SampleCount => _cachedRecords.Count;

    /// <inheritdoc/>
    protected override Task<(Tensor<T> Input, Tensor<T> Output)> ReadSampleAsync(
        int index,
        CancellationToken cancellationToken = default)
    {
        var record = _cachedRecords[index];
        var (features, labels) = _recordParser(record);
        return Task.FromResult((features, labels));
    }

    /// <inheritdoc/>
    protected override Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        _shardedDataset = new ShardedStreamingDataset(_shardPaths, _options);
        _cachedRecords = new List<byte[]>();

        foreach (var record in _shardedDataset.ReadRecords())
        {
            cancellationToken.ThrowIfCancellationRequested();
            _cachedRecords.Add(record);
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        _cachedRecords.Clear();
        _shardedDataset?.Dispose();
        _shardedDataset = null;
    }
}
