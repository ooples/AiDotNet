using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Validation;

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
        : base(batchSize, prefetchCount, numWorkers)
    {
        _sampleCount = sampleCount > 0 ? sampleCount : throw new ArgumentOutOfRangeException(nameof(sampleCount));
        Guard.NotNull(sampleReader);
        _sampleReader = sampleReader;
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
        : base(batchSize, prefetchCount, numWorkers)
    {
        if (string.IsNullOrWhiteSpace(directory))
        {
            throw new ArgumentNullException(nameof(directory));
        }

        Guard.NotNull(fileProcessor);
        _fileProcessor = fileProcessor;
        _filePaths = Directory.GetFiles(directory, filePattern, searchOption);

        if (_filePaths.Length == 0)
        {
            throw new ArgumentException($"No files matching pattern '{filePattern}' found in directory '{directory}'.", nameof(directory));
        }
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
    private readonly object _cacheLock = new object();
    private volatile string[]? _cachedLines;

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
        : base(batchSize, prefetchCount, numWorkers)
    {
        Guard.NotNullOrWhiteSpace(filePath);
        _filePath = filePath;
        Guard.NotNull(lineParser);
        _lineParser = lineParser;
        _hasHeader = hasHeader;

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"CSV file not found: {filePath}", filePath);
        }

        // Count lines (expensive but necessary for proper iteration)
        _lineCount = CountLines() - (hasHeader ? 1 : 0);
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
    /// <remarks>
    /// <para>
    /// For efficient random access, this method caches all lines on first access.
    /// This is a tradeoff: memory for speed. For truly streaming without random access,
    /// use <see cref="GetSequentialBatches"/> with shuffle=false.
    /// </para>
    /// <para>
    /// <b>Thread Safety:</b> This method uses double-checked locking to ensure
    /// thread-safe lazy initialization of the line cache. Multiple threads can
    /// safely call this method concurrently.
    /// </para>
    /// </remarks>
    protected override Task<(TInput Input, TOutput Output)> ReadSampleAsync(
        int index,
        CancellationToken cancellationToken = default)
    {
        // Thread-safe lazy initialization using double-checked locking pattern
        // First check without lock for fast path when cache is already populated
        if (_cachedLines == null)
        {
            lock (_cacheLock)
            {
                // Second check inside lock to prevent multiple threads from populating cache
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
            }
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
/// A streaming data loader that uses memory-mapped files for efficient random access
/// to large binary datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output/label data.</typeparam>
/// <remarks>
/// <para>
/// MemoryMappedStreamingDataLoader uses <see cref="System.IO.MemoryMappedFiles.MemoryMappedFile"/>
/// for efficient random access to large datasets stored in binary format. The operating system
/// handles paging data in and out of physical memory as needed, enabling efficient access to
/// datasets larger than available RAM.
/// </para>
/// <para>
/// <b>File Format Requirements:</b>
/// <list type="bullet">
/// <item><description>Binary file with fixed-size samples</description></item>
/// <item><description>Each sample is <c>inputSizeBytes + outputSizeBytes</c> bytes</description></item>
/// <item><description>Samples are stored contiguously with optional header</description></item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Memory-mapped files let the operating system manage
/// which parts of a large file are in memory. When you access a sample, the OS automatically
/// loads that portion of the file into RAM. This is very efficient for random access
/// patterns like shuffled batch iteration on datasets too large to fit in memory.
///
/// Example:
/// <code>
/// // Create a memory-mapped loader for binary image data
/// var loader = new MemoryMappedStreamingDataLoader&lt;float, float[], int&gt;(
///     filePath: "images.bin",
///     sampleCount: 60000,
///     inputSizeBytes: 784 * sizeof(float),   // 28x28 image
///     outputSizeBytes: sizeof(int),           // Label
///     inputDeserializer: (bytes) =&gt; {
///         var floats = new float[784];
///         for (int i = 0; i &lt; 784; i++)
///             floats[i] = BitConverter.ToSingle(bytes, i * 4);
///         return floats;
///     },
///     outputDeserializer: (bytes) =&gt; BitConverter.ToInt32(bytes, 0),
///     batchSize: 32
/// );
///
/// await foreach (var batch in loader.GetBatchesAsync())
/// {
///     await model.TrainOnBatchAsync(batch.Inputs, batch.Outputs);
/// }
/// </code>
/// </para>
/// </remarks>
public class MemoryMappedStreamingDataLoader<T, TInput, TOutput> : StreamingDataLoaderBase<T, TInput, TOutput>, IDisposable
{
    private readonly string _filePath;
    private readonly int _sampleCount;
    private readonly int _inputSizeBytes;
    private readonly int _outputSizeBytes;
    private readonly int _sampleSizeBytes;
    private readonly long _headerSizeBytes;
    private readonly Func<byte[], TInput> _inputDeserializer;
    private readonly Func<byte[], TOutput> _outputDeserializer;

    private MemoryMappedFile? _memoryMappedFile;
    private MemoryMappedViewAccessor? _viewAccessor;
    private readonly object _initLock = new object();
    private volatile bool _initialized;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the MemoryMappedStreamingDataLoader class.
    /// </summary>
    /// <param name="filePath">Path to the binary data file.</param>
    /// <param name="sampleCount">Total number of samples in the dataset.</param>
    /// <param name="inputSizeBytes">Size of input data per sample in bytes.</param>
    /// <param name="outputSizeBytes">Size of output/label data per sample in bytes.</param>
    /// <param name="inputDeserializer">Function to deserialize input bytes to TInput.</param>
    /// <param name="outputDeserializer">Function to deserialize output bytes to TOutput.</param>
    /// <param name="batchSize">Number of samples per batch.</param>
    /// <param name="headerSizeBytes">Size of file header to skip in bytes. Default is 0.</param>
    /// <param name="prefetchCount">Number of batches to prefetch. Default is 2.</param>
    /// <param name="numWorkers">Number of parallel workers. Default is 4.</param>
    /// <exception cref="ArgumentNullException">Thrown when filePath or deserializers are null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when sizes are invalid.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    public MemoryMappedStreamingDataLoader(
        string filePath,
        int sampleCount,
        int inputSizeBytes,
        int outputSizeBytes,
        Func<byte[], TInput> inputDeserializer,
        Func<byte[], TOutput> outputDeserializer,
        int batchSize,
        long headerSizeBytes = 0,
        int prefetchCount = 2,
        int numWorkers = 4)
        : base(batchSize, prefetchCount, numWorkers)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentNullException(nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Binary data file not found: {filePath}", filePath);
        }

        _filePath = filePath;
        _sampleCount = sampleCount > 0 ? sampleCount : throw new ArgumentOutOfRangeException(nameof(sampleCount), "Sample count must be positive.");
        _inputSizeBytes = inputSizeBytes > 0 ? inputSizeBytes : throw new ArgumentOutOfRangeException(nameof(inputSizeBytes), "Input size must be positive.");
        _outputSizeBytes = outputSizeBytes > 0 ? outputSizeBytes : throw new ArgumentOutOfRangeException(nameof(outputSizeBytes), "Output size must be positive.");
        _sampleSizeBytes = _inputSizeBytes + _outputSizeBytes;
        _headerSizeBytes = headerSizeBytes >= 0 ? headerSizeBytes : throw new ArgumentOutOfRangeException(nameof(headerSizeBytes), "Header size cannot be negative.");
        Guard.NotNull(inputDeserializer);
        _inputDeserializer = inputDeserializer;
        Guard.NotNull(outputDeserializer);
        _outputDeserializer = outputDeserializer;

        // Validate file size matches expected data size
        var fileInfo = new FileInfo(filePath);
        long expectedSize = _headerSizeBytes + ((long)_sampleCount * _sampleSizeBytes);
        if (fileInfo.Length < expectedSize)
        {
            throw new ArgumentException(
                $"File size ({fileInfo.Length} bytes) is smaller than expected ({expectedSize} bytes) for {sampleCount} samples of {_sampleSizeBytes} bytes each plus {headerSizeBytes} header bytes.",
                nameof(filePath));
        }
    }

    /// <inheritdoc/>
    public override string Name => "MemoryMappedStreamingDataLoader";

    /// <inheritdoc/>
    public override int SampleCount => _sampleCount;

    /// <summary>
    /// Gets the size of each sample in bytes (input + output).
    /// </summary>
    public int SampleSizeBytes => _sampleSizeBytes;

    /// <summary>
    /// Gets the size of the file header in bytes.
    /// </summary>
    public long HeaderSizeBytes => _headerSizeBytes;

    /// <summary>
    /// Gets the view accessor for reading from the memory-mapped file.
    /// Thread-safe initialization with proper null checking.
    /// </summary>
    /// <returns>The initialized view accessor.</returns>
    /// <exception cref="ObjectDisposedException">Thrown if the loader has been disposed.</exception>
    /// <exception cref="InvalidOperationException">Thrown if initialization fails.</exception>
    private MemoryMappedViewAccessor GetViewAccessor()
    {
        if (_initialized)
        {
            var accessor = _viewAccessor;
            if (accessor is not null)
            {
                return accessor;
            }
        }

        lock (_initLock)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(MemoryMappedStreamingDataLoader<T, TInput, TOutput>));
            }

            if (_initialized && _viewAccessor is not null)
            {
                return _viewAccessor;
            }

            // Create memory-mapped file with read-only access
            _memoryMappedFile = MemoryMappedFile.CreateFromFile(
                _filePath,
                FileMode.Open,
                mapName: null,
                capacity: 0,
                MemoryMappedFileAccess.Read);

            // Create a view accessor for the entire file
            _viewAccessor = _memoryMappedFile.CreateViewAccessor(
                offset: 0,
                size: 0,
                MemoryMappedFileAccess.Read);

            _initialized = true;

            return _viewAccessor;
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Reads a sample directly from the memory-mapped file. The operating system
    /// handles paging the data into memory as needed, making this efficient for
    /// datasets larger than available RAM.
    /// </remarks>
    protected override Task<(TInput Input, TOutput Output)> ReadSampleAsync(
        int index,
        CancellationToken cancellationToken = default)
    {
        if (index < 0 || index >= _sampleCount)
        {
            throw new ArgumentOutOfRangeException(nameof(index), $"Sample index {index} is out of range [0, {_sampleCount}).");
        }

        // Get the view accessor with proper null checking
        var viewAccessor = GetViewAccessor();

        // Calculate offset for this sample
        long offset = _headerSizeBytes + ((long)index * _sampleSizeBytes);

        // Read bytes for input and output
        byte[] inputBytes = new byte[_inputSizeBytes];
        byte[] outputBytes = new byte[_outputSizeBytes];

        // Read from memory-mapped file (thread-safe as we're reading from different positions)
        int inputBytesRead = viewAccessor.ReadArray(offset, inputBytes, 0, _inputSizeBytes);
        if (inputBytesRead != _inputSizeBytes)
        {
            throw new InvalidOperationException(
                $"Failed to read input data for sample {index}. Expected {_inputSizeBytes} bytes, got {inputBytesRead}.");
        }

        int outputBytesRead = viewAccessor.ReadArray(offset + _inputSizeBytes, outputBytes, 0, _outputSizeBytes);
        if (outputBytesRead != _outputSizeBytes)
        {
            throw new InvalidOperationException(
                $"Failed to read output data for sample {index}. Expected {_outputSizeBytes} bytes, got {outputBytesRead}.");
        }

        // Deserialize
        TInput input = _inputDeserializer(inputBytes);
        TOutput output = _outputDeserializer(outputBytes);

        return Task.FromResult((input, output));
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        // Don't unload memory-mapped file - let OS manage the pages
        base.UnloadDataCore();
    }

    /// <summary>
    /// Releases all resources used by the memory-mapped data loader.
    /// </summary>
    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases the unmanaged resources and optionally releases the managed resources.
    /// </summary>
    /// <param name="disposing">True to release both managed and unmanaged resources.</param>
    protected void Dispose(bool disposing)
    {
        if (_disposed)
        {
            return;
        }

        if (disposing)
        {
            lock (_initLock)
            {
                if (_viewAccessor is not null)
                {
                    _viewAccessor.Dispose();
                    _viewAccessor = null;
                }

                if (_memoryMappedFile is not null)
                {
                    _memoryMappedFile.Dispose();
                    _memoryMappedFile = null;
                }
            }
        }

        _disposed = true;
    }

    /// <summary>
    /// Finalizer to ensure resources are released.
    /// </summary>
    ~MemoryMappedStreamingDataLoader()
    {
        Dispose(disposing: false);
    }
}
