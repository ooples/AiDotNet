using AiDotNet.Data.Loaders;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Formats;

/// <summary>
/// A typed data loader facade that wraps <see cref="WebDataset"/> and implements
/// <see cref="StreamingDataLoaderBase{T, TInput, TOutput}"/> for IDataLoader compliance.
/// </summary>
/// <remarks>
/// <para>
/// This loader eagerly reads all samples from the underlying TAR archives during loading,
/// caches them in memory, and provides index-based access via a user-supplied parser delegate
/// that converts raw <c>Dictionary&lt;string, byte[]&gt;</c> samples into typed tensor pairs.
/// </para>
/// <para><b>For Beginners:</b> Use this when you want to load WebDataset TAR archives and
/// feed them into <c>AiModelBuilder.ConfigureDataLoader()</c>. You provide a parser function
/// that knows how to decode the raw bytes (images, labels, etc.) into tensors.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for tensor values.</typeparam>
public class WebDatasetDataLoader<T> : StreamingDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly string[] _tarPaths;
    private readonly Func<Dictionary<string, byte[]>, (Tensor<T>, Tensor<T>)> _sampleParser;
    private readonly WebDatasetOptions? _options;
    private List<Dictionary<string, byte[]>> _cachedSamples = new();
    private WebDataset? _webDataset;

    /// <summary>
    /// Creates a new WebDatasetDataLoader.
    /// </summary>
    /// <param name="tarPaths">Paths to the TAR archive files (shards).</param>
    /// <param name="sampleParser">
    /// A function that converts a raw sample (dictionary of extension to bytes) into a
    /// (features, labels) tensor pair.
    /// </param>
    /// <param name="batchSize">Number of samples per batch. Default is 32.</param>
    /// <param name="options">Optional WebDataset configuration.</param>
    public WebDatasetDataLoader(
        string[] tarPaths,
        Func<Dictionary<string, byte[]>, (Tensor<T>, Tensor<T>)> sampleParser,
        int batchSize = 32,
        WebDatasetOptions? options = null)
        : base(batchSize)
    {
        if (tarPaths is null || tarPaths.Length == 0)
            throw new ArgumentException("At least one TAR path is required.", nameof(tarPaths));
        if (sampleParser is null)
            throw new ArgumentNullException(nameof(sampleParser));

        _tarPaths = tarPaths;
        _sampleParser = sampleParser;
        _options = options;
    }

    /// <inheritdoc/>
    public override string Name => "WebDataset";

    /// <inheritdoc/>
    public override int SampleCount => _cachedSamples.Count;

    /// <inheritdoc/>
    protected override Task<(Tensor<T> Input, Tensor<T> Output)> ReadSampleAsync(
        int index,
        CancellationToken cancellationToken = default)
    {
        var rawSample = _cachedSamples[index];
        var (features, labels) = _sampleParser(rawSample);
        return Task.FromResult((features, labels));
    }

    /// <inheritdoc/>
    protected override Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        _webDataset = new WebDataset(_tarPaths, _options);
        _cachedSamples = new List<Dictionary<string, byte[]>>();

        foreach (var sample in _webDataset.ReadSamples())
        {
            cancellationToken.ThrowIfCancellationRequested();
            _cachedSamples.Add(sample);
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        _cachedSamples.Clear();
        _webDataset?.Dispose();
        _webDataset = null;
    }
}
