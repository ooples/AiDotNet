using System.Runtime.CompilerServices;
using System.Threading.Channels;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// A data loader that streams data from disk or other sources without loading all data into memory.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output/label data.</typeparam>
/// <remarks>
/// <para>
/// StreamingDataLoader is designed for datasets that don't fit in memory. Instead of loading
/// all data upfront, it reads data on-demand from a source, processes it, and yields batches.
/// </para>
/// <para><b>For Beginners:</b> When your dataset is too large to fit in RAM (e.g., millions
/// of images or text documents), you can't load it all at once. StreamingDataLoader solves
/// this by reading data piece by piece as needed.
///
/// Example:
/// <code>
/// // Define how to read individual samples
/// var loader = new StreamingDataLoader&lt;float, Tensor&lt;float&gt;, int&gt;(
///     sampleCount: 1000000,  // 1 million samples
///     sampleReader: async (index, ct) =&gt;
///     {
///         var image = await LoadImageFromDisk(index, ct);
///         var label = await LoadLabelFromDisk(index, ct);
///         return (image, label);
///     },
///     batchSize: 32
/// );
///
/// await foreach (var (inputs, labels) in loader.GetBatchesAsync())
/// {
///     await model.TrainOnBatchAsync(inputs, labels);
/// }
/// </code>
/// </para>
/// </remarks>
public class StreamingDataLoader<T, TInput, TOutput> : StreamingDataLoaderBase<T, TInput, TOutput>
{
    private readonly int _sampleCount;
    private readonly Func<int, CancellationToken, Task<(TInput, TOutput)>> _sampleReader;
    private readonly string _name;

    /// <summary>
    /// Initializes a new instance of the StreamingDataLoader class.
    /// </summary>
    /// <param name="sampleCount">Total number of samples in the dataset.</param>
    /// <param name="sampleReader">Async function that reads a single sample by index.</param>
    /// <param name="batchSize">Number of samples per batch.</param>
    /// <param name="name">Optional name for the data loader.</param>
    /// <param name="prefetchCount">Number of batches to prefetch. Default is 2.</param>
    /// <param name="numWorkers">Number of parallel workers for sample loading. Default is 4.</param>
    public StreamingDataLoader(
        int sampleCount,
        Func<int, CancellationToken, Task<(TInput, TOutput)>> sampleReader,
        int batchSize,
        string? name = null,
        int prefetchCount = 2,
        int numWorkers = 4)
        : base(prefetchCount, numWorkers)
    {
        _sampleCount = sampleCount > 0 ? sampleCount : throw new ArgumentOutOfRangeException(nameof(sampleCount));
        _sampleReader = sampleReader ?? throw new ArgumentNullException(nameof(sampleReader));
        BatchSize = batchSize > 0 ? batchSize : throw new ArgumentOutOfRangeException(nameof(batchSize));
        _name = name ?? "StreamingDataLoader";
    }

    /// <inheritdoc/>
    public override string Name => _name;

    /// <inheritdoc/>
    public override int SampleCount => _sampleCount;

    /// <inheritdoc/>
    protected override Task<(TInput Input, TOutput Output)> ReadSampleAsync(
        int index,
        CancellationToken cancellationToken = default)
    {
        return _sampleReader(index, cancellationToken);
    }
}

/// <summary>
/// A streaming data loader that reads from files in a directory.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output/label data.</typeparam>
/// <remarks>
/// <para>
/// FileStreamingDataLoader automatically discovers files in a directory and streams them
/// during training. This is ideal for image datasets where each file is a sample.
/// </para>
/// <para><b>For Beginners:</b> If you have a folder full of images with labels in the filename
/// or a separate label file, this loader will read them one by one as needed.
///
/// Example:
/// <code>
/// var loader = new FileStreamingDataLoader&lt;float, float[], int&gt;(
///     directory: "path/to/images",
///     filePattern: "*.png",
///     fileProcessor: async (filePath, ct) =&gt;
///     {
///         var pixels = await LoadImagePixels(filePath, ct);
///         var label = ExtractLabelFromPath(filePath);
///         return (pixels, label);
///     },
///     batchSize: 32
/// );
/// </code>
/// </para>
/// </remarks>
public class FileStreamingDataLoader<T, TInput, TOutput> : StreamingDataLoaderBase<T, TInput, TOutput>
{
    private readonly string[] _filePaths;
    private readonly Func<string, CancellationToken, Task<(TInput, TOutput)>> _fileProcessor;

    /// <summary>
    /// Initializes a new instance of the FileStreamingDataLoader class.
    /// </summary>
    /// <param name="directory">The directory containing the data files.</param>
    /// <param name="filePattern">The file pattern to match (e.g., "*.png").</param>
    /// <param name="fileProcessor">Function that processes a file and returns (input, output).</param>
    /// <param name="batchSize">Number of samples per batch.</param>
    /// <param name="searchOption">Whether to search subdirectories.</param>
    /// <param name="prefetchCount">Number of batches to prefetch.</param>
    /// <param name="numWorkers">Number of parallel workers.</param>
    public FileStreamingDataLoader(
        string directory,
        string filePattern,
        Func<string, CancellationToken, Task<(TInput, TOutput)>> fileProcessor,
        int batchSize,
        SearchOption searchOption = SearchOption.TopDirectoryOnly,
        int prefetchCount = 2,
        int numWorkers = 4)
        : base(prefetchCount, numWorkers)
    {
        if (string.IsNullOrWhiteSpace(directory))
        {
            throw new ArgumentNullException(nameof(directory));
        }

        _fileProcessor = fileProcessor ?? throw new ArgumentNullException(nameof(fileProcessor));
        _filePaths = Directory.GetFiles(directory, filePattern, searchOption);

        if (_filePaths.Length == 0)
        {
            throw new ArgumentException($"No files matching pattern '{filePattern}' found in directory '{directory}'.", nameof(directory));
        }

        BatchSize = batchSize > 0 ? batchSize : throw new ArgumentOutOfRangeException(nameof(batchSize));
    }

    /// <inheritdoc/>
    public override string Name => "FileStreamingDataLoader";

    /// <inheritdoc/>
    public override int SampleCount => _filePaths.Length;

    /// <summary>
    /// Gets all file paths in the dataset.
    /// </summary>
    public IReadOnlyList<string> FilePaths => _filePaths;

    /// <inheritdoc/>
    protected override Task<(TInput Input, TOutput Output)> ReadSampleAsync(
        int index,
        CancellationToken cancellationToken = default)
    {
        return _fileProcessor(_filePaths[index], cancellationToken);
    }
}

/// <summary>
/// A streaming data loader that reads from a CSV file line by line.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output/label data.</typeparam>
/// <remarks>
/// <para>
/// CsvStreamingDataLoader reads a CSV file line by line without loading the entire file
/// into memory. This is ideal for large tabular datasets.
/// </para>
/// <para><b>For Beginners:</b> If you have a large CSV file (gigabytes of data), this loader
/// will read it row by row as needed during training.
///
/// Example:
/// <code>
/// var loader = new CsvStreamingDataLoader&lt;float, float[], float&gt;(
///     filePath: "large_dataset.csv",
///     lineParser: (line, lineNumber) =&gt;
///     {
///         var parts = line.Split(',');
///         var features = parts.Take(10).Select(float.Parse).ToArray();
///         var label = float.Parse(parts[10]);
///         return (features, label);
///     },
///     batchSize: 256,
///     hasHeader: true
/// );
/// </code>
/// </para>
/// </remarks>
public class CsvStreamingDataLoader<T, TInput, TOutput> : StreamingDataLoaderBase<T, TInput, TOutput>
{
    private readonly string _filePath;
    private readonly Func<string, int, (TInput, TOutput)> _lineParser;
    private readonly bool _hasHeader;
    private readonly int _lineCount;
    private string[]? _cachedLines;

    /// <summary>
    /// Initializes a new instance of the CsvStreamingDataLoader class.
    /// </summary>
    /// <param name="filePath">Path to the CSV file.</param>
    /// <param name="lineParser">Function that parses a line into (input, output).</param>
    /// <param name="batchSize">Number of samples per batch.</param>
    /// <param name="hasHeader">Whether the CSV has a header row to skip.</param>
    /// <param name="prefetchCount">Number of batches to prefetch.</param>
    /// <param name="numWorkers">Number of parallel workers.</param>
    public CsvStreamingDataLoader(
        string filePath,
        Func<string, int, (TInput, TOutput)> lineParser,
        int batchSize,
        bool hasHeader = true,
        int prefetchCount = 2,
        int numWorkers = 4)
        : base(prefetchCount, numWorkers)
    {
        _filePath = filePath ?? throw new ArgumentNullException(nameof(filePath));
        _lineParser = lineParser ?? throw new ArgumentNullException(nameof(lineParser));
        _hasHeader = hasHeader;

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"CSV file not found: {filePath}", filePath);
        }

        // Count lines (expensive but necessary for proper iteration)
        _lineCount = CountLines() - (hasHeader ? 1 : 0);
        BatchSize = batchSize > 0 ? batchSize : throw new ArgumentOutOfRangeException(nameof(batchSize));
    }

    private int CountLines()
    {
        int count = 0;
        using var reader = new StreamReader(_filePath);
        while (reader.ReadLine() != null)
        {
            count++;
        }
        return count;
    }

    /// <inheritdoc/>
    public override string Name => "CsvStreamingDataLoader";

    /// <inheritdoc/>
    public override int SampleCount => _lineCount;

    /// <inheritdoc/>
    protected override Task<(TInput Input, TOutput Output)> ReadSampleAsync(
        int index,
        CancellationToken cancellationToken = default)
    {
        // For efficient random access, we cache all lines on first access
        // This is a tradeoff: memory for speed. For truly streaming without
        // random access, use sequential iteration with shuffle=false.
        if (_cachedLines == null)
        {
            var lines = new List<string>(_lineCount);
            using (var reader = new StreamReader(_filePath))
            {
                if (_hasHeader)
                {
                    reader.ReadLine();
                }

                string? line;
                while ((line = reader.ReadLine()) != null)
                {
                    lines.Add(line);
                }
            }
            _cachedLines = lines.ToArray();
        }

        var result = _lineParser(_cachedLines[index], index);
        return Task.FromResult(result);
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        _cachedLines = null;
        base.UnloadDataCore();
    }

    /// <summary>
    /// Iterates through the CSV file sequentially without loading all lines into memory.
    /// </summary>
    /// <param name="batchSize">Batch size to use.</param>
    /// <param name="dropLast">Whether to drop the last incomplete batch.</param>
    /// <returns>An enumerable of batches.</returns>
    /// <remarks>
    /// This method provides true streaming iteration without caching all lines.
    /// Use this when memory is constrained and you don't need shuffling.
    /// </remarks>
    public IEnumerable<(TInput[] Inputs, TOutput[] Outputs)> GetSequentialBatches(
        int? batchSize = null,
        bool dropLast = false)
    {
        int actualBatchSize = batchSize ?? BatchSize;

        using var reader = new StreamReader(_filePath);

        // Skip header if present
        if (_hasHeader)
        {
            reader.ReadLine();
        }

        var inputBatch = new List<TInput>(actualBatchSize);
        var outputBatch = new List<TOutput>(actualBatchSize);
        int lineNumber = 0;
        string? line;

        while ((line = reader.ReadLine()) != null)
        {
            var (input, output) = _lineParser(line, lineNumber);
            inputBatch.Add(input);
            outputBatch.Add(output);
            lineNumber++;

            if (inputBatch.Count == actualBatchSize)
            {
                yield return (inputBatch.ToArray(), outputBatch.ToArray());
                inputBatch.Clear();
                outputBatch.Clear();
            }
        }

        // Handle remaining samples
        if (inputBatch.Count > 0 && !dropLast)
        {
            yield return (inputBatch.ToArray(), outputBatch.ToArray());
        }
    }
}

/// <summary>
/// A streaming data loader that uses memory-mapped files for efficient random access.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output/label data.</typeparam>
/// <remarks>
/// <para>
/// MemoryMappedStreamingDataLoader uses memory-mapped files for efficient random access
/// to large datasets stored in binary format. The operating system handles paging data
/// in and out of memory as needed.
/// </para>
/// <para><b>For Beginners:</b> Memory-mapped files let the operating system manage
/// which parts of a large file are in memory. This is very efficient for random access
/// patterns like shuffled batch iteration.
/// </para>
/// </remarks>
public class MemoryMappedStreamingDataLoader<T, TInput, TOutput> : StreamingDataLoader<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the MemoryMappedStreamingDataLoader class.
    /// </summary>
    /// <param name="sampleCount">Total number of samples in the dataset.</param>
    /// <param name="sampleReader">Async function that reads a sample by index.</param>
    /// <param name="batchSize">Number of samples per batch.</param>
    /// <param name="prefetchCount">Number of batches to prefetch.</param>
    /// <param name="numWorkers">Number of parallel workers.</param>
    public MemoryMappedStreamingDataLoader(
        int sampleCount,
        Func<int, CancellationToken, Task<(TInput, TOutput)>> sampleReader,
        int batchSize,
        int prefetchCount = 2,
        int numWorkers = 4)
        : base(sampleCount, sampleReader, batchSize, "MemoryMappedStreamingDataLoader", prefetchCount, numWorkers)
    {
    }
}
