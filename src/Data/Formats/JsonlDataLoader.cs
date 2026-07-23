using AiDotNet.Data.Loaders;
using AiDotNet.LinearAlgebra;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Data.Formats;

/// <summary>
/// A typed data loader facade that wraps <see cref="JsonlStreamingLoader"/> and implements
/// <see cref="StreamingDataLoaderBase{T, TInput, TOutput}"/> for IDataLoader compliance.
/// </summary>
/// <remarks>
/// <para>
/// This loader eagerly reads all JSON objects from the underlying JSONL files during loading,
/// caches them in memory, and provides index-based access via a user-supplied parser delegate
/// that converts <see cref="JObject"/> records into typed tensor pairs.
/// </para>
/// <para><b>For Beginners:</b> Use this when you want to load JSONL files and feed them into
/// <c>AiModelBuilder.ConfigureDataLoader()</c>. You provide a parser function that knows how
/// to convert each JSON record into tensors for training.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for tensor values.</typeparam>
public class JsonlDataLoader<T> : StreamingDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly string[] _filePaths;
    private readonly Func<JObject, (Tensor<T>, Tensor<T>)> _recordParser;
    private readonly string? _textField;
    private readonly string? _labelField;
    private readonly int _shuffleBufferSize;
    private List<JObject> _cachedRecords = new();
    private JsonlStreamingLoader? _jsonlLoader;

    /// <summary>
    /// Creates a new JsonlDataLoader.
    /// </summary>
    /// <param name="filePaths">Paths to the JSONL files.</param>
    /// <param name="recordParser">
    /// A function that converts a parsed <see cref="JObject"/> into a
    /// (features, labels) tensor pair.
    /// </param>
    /// <param name="batchSize">Number of samples per batch. Default is 32.</param>
    /// <param name="textField">Optional JSON field name for text data.</param>
    /// <param name="labelField">Optional JSON field name for labels.</param>
    /// <param name="shuffleBufferSize">Buffer size for shuffling (0 = no shuffle).</param>
    public JsonlDataLoader(
        string[] filePaths,
        Func<JObject, (Tensor<T>, Tensor<T>)> recordParser,
        int batchSize = 32,
        string? textField = null,
        string? labelField = null,
        int shuffleBufferSize = 0)
        : base(batchSize)
    {
        if (filePaths is null || filePaths.Length == 0)
            throw new ArgumentException("At least one file path is required.", nameof(filePaths));
        if (recordParser is null)
            throw new ArgumentNullException(nameof(recordParser));

        _filePaths = filePaths;
        _recordParser = recordParser;
        _textField = textField;
        _labelField = labelField;
        _shuffleBufferSize = shuffleBufferSize;
    }

    /// <inheritdoc/>
    public override string Name => "JSONL";

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
        _jsonlLoader = new JsonlStreamingLoader(_filePaths, _textField, _labelField, _shuffleBufferSize);
        _cachedRecords = new List<JObject>();

        try
        {
            foreach (var obj in _jsonlLoader.ReadObjects())
            {
                cancellationToken.ThrowIfCancellationRequested();
                _cachedRecords.Add(obj);
            }
        }
        catch
        {
            _cachedRecords.Clear();
            _jsonlLoader?.Dispose();
            _jsonlLoader = null;
            throw;
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        _cachedRecords.Clear();
        _jsonlLoader?.Dispose();
        _jsonlLoader = null;
    }
}
