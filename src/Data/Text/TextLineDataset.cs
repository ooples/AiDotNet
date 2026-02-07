using AiDotNet.Data.Loaders;

namespace AiDotNet.Data.Text;

/// <summary>
/// Streams text line-by-line from a file for language modeling tasks.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Reads a text file line by line, providing each line as a string sample.
/// Extends DataLoaderBase for lifecycle management while exposing line-reading
/// as an async enumerable.
/// </para>
/// <para><b>For Beginners:</b> This loader is for large text files that you want to
/// process line by line. Each line becomes one sample in your dataset.
/// </para>
/// </remarks>
public class TextLineDataset<T> : DataLoaderBase<T>
{
    private readonly string _filePath;
    private readonly bool _skipEmptyLines;
    private int _lineCount;

    /// <inheritdoc/>
    public override string Name => "TextLine";

    /// <inheritdoc/>
    public override string Description => $"Text line dataset from {Path.GetFileName(_filePath)}";

    /// <inheritdoc/>
    public override int TotalCount => _lineCount;

    /// <summary>
    /// Gets the file path being read.
    /// </summary>
    public string FilePath => _filePath;

    /// <summary>
    /// Creates a new TextLineDataset that reads from the specified file.
    /// </summary>
    /// <param name="filePath">Path to the text file.</param>
    /// <param name="skipEmptyLines">Whether to skip empty lines. Default is true.</param>
    /// <param name="batchSize">Batch size for iteration. Default is 32.</param>
    public TextLineDataset(string filePath, bool skipEmptyLines = true, int batchSize = 32)
        : base(batchSize)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentNullException(nameof(filePath));
        }

        _filePath = filePath;
        _skipEmptyLines = skipEmptyLines;
    }

    /// <inheritdoc/>
    protected override Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        if (!File.Exists(_filePath))
        {
            throw new FileNotFoundException($"Text file not found: {_filePath}");
        }

        // Count lines for TotalCount
        _lineCount = 0;
        using (var reader = new StreamReader(_filePath))
        {
            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!_skipEmptyLines || line.Trim().Length > 0)
                {
                    _lineCount++;
                }
            }
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        _lineCount = 0;
    }

    /// <summary>
    /// Reads all lines from the text file.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>An async enumerable of text lines.</returns>
    public async IAsyncEnumerable<string> ReadLinesAsync(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        EnsureLoaded();

        using var reader = new StreamReader(_filePath);
        string? line;
        while ((line = await reader.ReadLineAsync()) is not null)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!_skipEmptyLines || line.Trim().Length > 0)
            {
                yield return line;
            }
        }
    }

    /// <summary>
    /// Reads lines in batches.
    /// </summary>
    /// <param name="batchSize">Number of lines per batch. Uses default BatchSize if null.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>An async enumerable of string array batches.</returns>
    public async IAsyncEnumerable<string[]> ReadBatchesAsync(
        int? batchSize = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        EnsureLoaded();

        int effectiveBatchSize = batchSize ?? BatchSize;
        var batch = new List<string>(effectiveBatchSize);

        await foreach (var line in ReadLinesAsync(cancellationToken))
        {
            batch.Add(line);
            if (batch.Count >= effectiveBatchSize)
            {
                yield return batch.ToArray();
                batch.Clear();
            }
        }

        if (batch.Count > 0)
        {
            yield return batch.ToArray();
        }
    }
}
