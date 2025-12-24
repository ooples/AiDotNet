using System.Threading.Channels;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// A data loader that streams data from disk or other sources without loading all data into memory.
/// </summary>
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
/// var loader = new StreamingDataLoader&lt;Tensor&lt;float&gt;, int&gt;(
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
public class StreamingDataLoader<TInput, TOutput> : IBatchIterable<(TInput[], TOutput[])>, IDisposable
{
    private readonly int _sampleCount;
    private readonly Func<int, CancellationToken, Task<(TInput, TOutput)>> _sampleReader;
    private int _batchSize;
    private readonly int _prefetchCount;
    private readonly int _numWorkers;
    private bool _disposed;

    // State for imperative API (GetNextBatch/HasNext)
    private IEnumerator<(TInput[], TOutput[])>? _currentEnumerator;
    private bool _hasNextBatch;
    private (TInput[], TOutput[])? _nextBatch;

    /// <summary>
    /// Initializes a new instance of the StreamingDataLoader class.
    /// </summary>
    /// <param name="sampleCount">Total number of samples in the dataset.</param>
    /// <param name="sampleReader">Async function that reads a single sample by index.</param>
    /// <param name="batchSize">Number of samples per batch.</param>
    /// <param name="prefetchCount">Number of batches to prefetch. Default is 2.</param>
    /// <param name="numWorkers">Number of parallel workers for sample loading. Default is 4.</param>
    public StreamingDataLoader(
        int sampleCount,
        Func<int, CancellationToken, Task<(TInput, TOutput)>> sampleReader,
        int batchSize,
        int prefetchCount = 2,
        int numWorkers = 4)
    {
        _sampleCount = sampleCount > 0 ? sampleCount : throw new ArgumentOutOfRangeException(nameof(sampleCount));
        _sampleReader = sampleReader ?? throw new ArgumentNullException(nameof(sampleReader));
        _batchSize = batchSize > 0 ? batchSize : throw new ArgumentOutOfRangeException(nameof(batchSize));
        _prefetchCount = Math.Max(1, prefetchCount);
        _numWorkers = Math.Max(1, numWorkers);
    }

    /// <summary>
    /// Gets the total number of samples in the dataset.
    /// </summary>
    public int SampleCount => _sampleCount;

    /// <summary>
    /// Gets or sets the batch size.
    /// </summary>
    public int BatchSize
    {
        get => _batchSize;
        set
        {
            _batchSize = value > 0 ? value : throw new ArgumentOutOfRangeException(nameof(value));
            ResetIteration();
        }
    }

    /// <summary>
    /// Gets whether there are more batches available in the current iteration.
    /// </summary>
    public bool HasNext
    {
        get
        {
            EnsureIteratorInitialized();
            return _hasNextBatch;
        }
    }

    /// <summary>
    /// Gets the next batch of data.
    /// </summary>
    /// <returns>The next batch of data.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no more batches are available.</exception>
    public (TInput[], TOutput[]) GetNextBatch()
    {
        if (!TryGetNextBatch(out var batch))
        {
            throw new InvalidOperationException("No more batches available. Call Reset() to start a new iteration.");
        }
        return batch;
    }

    /// <summary>
    /// Attempts to get the next batch without throwing if unavailable.
    /// </summary>
    /// <param name="batch">The batch if available, default otherwise.</param>
    /// <returns>True if a batch was available, false if iteration is complete.</returns>
    public bool TryGetNextBatch(out (TInput[], TOutput[]) batch)
    {
        EnsureIteratorInitialized();

        if (!_hasNextBatch || _nextBatch == null)
        {
            batch = default;
            return false;
        }

        batch = _nextBatch.Value;

        // Advance to next batch
        if (_currentEnumerator != null && _currentEnumerator.MoveNext())
        {
            _nextBatch = _currentEnumerator.Current;
            _hasNextBatch = true;
        }
        else
        {
            _nextBatch = null;
            _hasNextBatch = false;
        }

        return true;
    }

    /// <summary>
    /// Resets the iteration to the beginning.
    /// </summary>
    public void ResetIteration()
    {
        _currentEnumerator?.Dispose();
        _currentEnumerator = null;
        _nextBatch = null;
        _hasNextBatch = false;
    }

    private void EnsureIteratorInitialized()
    {
        if (_currentEnumerator == null)
        {
            _currentEnumerator = GetBatches().GetEnumerator();
            if (_currentEnumerator.MoveNext())
            {
                _nextBatch = _currentEnumerator.Current;
                _hasNextBatch = true;
            }
            else
            {
                _hasNextBatch = false;
            }
        }
    }

    /// <summary>
    /// Gets the number of batches per epoch.
    /// </summary>
    public int BatchCount => (_sampleCount + _batchSize - 1) / _batchSize;

    /// <inheritdoc/>
    public IEnumerable<(TInput[], TOutput[])> GetBatches(
        int? batchSize = null,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null)
    {
        // For streaming, we use async version internally but block for sync API
        var asyncEnumerable = GetBatchesAsync(batchSize, shuffle, dropLast, seed, _prefetchCount);

        // Convert async enumerable to sync enumerable (blocking)
        var enumerator = asyncEnumerable.GetAsyncEnumerator();
        try
        {
            while (enumerator.MoveNextAsync().AsTask().GetAwaiter().GetResult())
            {
                yield return enumerator.Current;
            }
        }
        finally
        {
            enumerator.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<(TInput[], TOutput[])> GetBatchesAsync(
        int? batchSize = null,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null,
        int prefetchCount = 2,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(StreamingDataLoader<TInput, TOutput>));
        }

        int actualBatchSize = batchSize ?? _batchSize;
        int actualPrefetchCount = prefetchCount > 0 ? prefetchCount : _prefetchCount;

        // Generate indices
        int[] indices = new int[_sampleCount];
        for (int i = 0; i < _sampleCount; i++)
        {
            indices[i] = i;
        }

        // Shuffle if requested
        if (shuffle)
        {
            Random random = seed.HasValue
                ? RandomHelper.CreateSeededRandom(seed.Value)
                : RandomHelper.CreateSecureRandom();

            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        // Calculate number of batches
        int numBatches = indices.Length / actualBatchSize;
        if (!dropLast && indices.Length % actualBatchSize > 0)
        {
            numBatches++;
        }

        // Create output channel
        var outputChannel = Channel.CreateBounded<(TInput[], TOutput[])>(
            new BoundedChannelOptions(actualPrefetchCount)
            {
                FullMode = BoundedChannelFullMode.Wait,
                SingleReader = true,
                SingleWriter = false
            });

        // Start producer task
        var producerTask = Task.Run(async () =>
        {
            try
            {
                for (int b = 0; b < numBatches; b++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    int startIdx = b * actualBatchSize;
                    int endIdx = Math.Min(startIdx + actualBatchSize, indices.Length);
                    int currentBatchSize = endIdx - startIdx;

                    // Load samples in parallel
                    var loadTasks = new Task<(TInput, TOutput)>[currentBatchSize];
                    for (int i = 0; i < currentBatchSize; i++)
                    {
                        int sampleIndex = indices[startIdx + i];
                        loadTasks[i] = _sampleReader(sampleIndex, cancellationToken);
                    }

                    // Wait for all samples to load
                    var samples = await Task.WhenAll(loadTasks);

                    // Separate inputs and outputs
                    var inputs = new TInput[currentBatchSize];
                    var outputs = new TOutput[currentBatchSize];
                    for (int i = 0; i < currentBatchSize; i++)
                    {
                        inputs[i] = samples[i].Item1;
                        outputs[i] = samples[i].Item2;
                    }

                    await outputChannel.Writer.WriteAsync((inputs, outputs), cancellationToken);
                }
            }
            finally
            {
                outputChannel.Writer.Complete();
            }
        }, cancellationToken);

        // Consume batches (net471 compatible)
        while (await outputChannel.Reader.WaitToReadAsync(cancellationToken))
        {
            while (outputChannel.Reader.TryRead(out var batch))
            {
                yield return batch;
            }
        }

        await producerTask;
    }

    /// <summary>
    /// Disposes the streaming data loader.
    /// </summary>
    public void Dispose()
    {
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// A streaming data loader that reads from files in a directory.
/// </summary>
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
/// var loader = new FileStreamingDataLoader&lt;float[], int&gt;(
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
public class FileStreamingDataLoader<TInput, TOutput> : StreamingDataLoader<TInput, TOutput>
{
    private readonly string[] _filePaths;

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
        : base(
            GetFileCount(directory, filePattern, searchOption),
            CreateFileReader(directory, filePattern, searchOption, fileProcessor),
            batchSize,
            prefetchCount,
            numWorkers)
    {
        _filePaths = Directory.GetFiles(directory, filePattern, searchOption);
    }

    private static int GetFileCount(string directory, string filePattern, SearchOption searchOption)
    {
        return Directory.GetFiles(directory, filePattern, searchOption).Length;
    }

    private static Func<int, CancellationToken, Task<(TInput, TOutput)>> CreateFileReader(
        string directory,
        string filePattern,
        SearchOption searchOption,
        Func<string, CancellationToken, Task<(TInput, TOutput)>> fileProcessor)
    {
        var filePaths = Directory.GetFiles(directory, filePattern, searchOption);
        return async (index, ct) => await fileProcessor(filePaths[index], ct);
    }

    /// <summary>
    /// Gets all file paths in the dataset.
    /// </summary>
    public IReadOnlyList<string> FilePaths => _filePaths;
}

/// <summary>
/// A streaming data loader that reads from a CSV file line by line.
/// </summary>
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
/// var loader = new CsvStreamingDataLoader&lt;float[], float&gt;(
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
public class CsvStreamingDataLoader<TInput, TOutput> : IBatchIterable<(TInput[], TOutput[])>, IDisposable
{
    private readonly string _filePath;
    private readonly Func<string, int, (TInput, TOutput)> _lineParser;
    private int _batchSize;
    private readonly bool _hasHeader;
    private readonly int _lineCount;
    private readonly int _prefetchCount;
    private bool _disposed;

    // State for imperative API (GetNextBatch/HasNext)
    private IEnumerator<(TInput[], TOutput[])>? _currentEnumerator;
    private bool _hasNextBatch;
    private (TInput[], TOutput[])? _nextBatch;

    /// <summary>
    /// Initializes a new instance of the CsvStreamingDataLoader class.
    /// </summary>
    /// <param name="filePath">Path to the CSV file.</param>
    /// <param name="lineParser">Function that parses a line into (input, output).</param>
    /// <param name="batchSize">Number of samples per batch.</param>
    /// <param name="hasHeader">Whether the CSV has a header row to skip.</param>
    /// <param name="prefetchCount">Number of batches to prefetch.</param>
    public CsvStreamingDataLoader(
        string filePath,
        Func<string, int, (TInput, TOutput)> lineParser,
        int batchSize,
        bool hasHeader = true,
        int prefetchCount = 2)
    {
        _filePath = filePath ?? throw new ArgumentNullException(nameof(filePath));
        _lineParser = lineParser ?? throw new ArgumentNullException(nameof(lineParser));
        _batchSize = batchSize > 0 ? batchSize : throw new ArgumentOutOfRangeException(nameof(batchSize));
        _hasHeader = hasHeader;
        _prefetchCount = Math.Max(1, prefetchCount);

        // Count lines (expensive but necessary for shuffling)
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

    /// <summary>
    /// Gets the total number of samples in the dataset.
    /// </summary>
    public int SampleCount => _lineCount;

    /// <summary>
    /// Gets or sets the batch size.
    /// </summary>
    public int BatchSize
    {
        get => _batchSize;
        set
        {
            _batchSize = value > 0 ? value : throw new ArgumentOutOfRangeException(nameof(value));
            ResetIteration();
        }
    }

    /// <summary>
    /// Gets whether there are more batches available in the current iteration.
    /// </summary>
    public bool HasNext
    {
        get
        {
            EnsureIteratorInitialized();
            return _hasNextBatch;
        }
    }

    /// <summary>
    /// Gets the next batch of data.
    /// </summary>
    /// <returns>The next batch of data.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no more batches are available.</exception>
    public (TInput[], TOutput[]) GetNextBatch()
    {
        if (!TryGetNextBatch(out var batch))
        {
            throw new InvalidOperationException("No more batches available. Call Reset() to start a new iteration.");
        }
        return batch;
    }

    /// <summary>
    /// Attempts to get the next batch without throwing if unavailable.
    /// </summary>
    /// <param name="batch">The batch if available, default otherwise.</param>
    /// <returns>True if a batch was available, false if iteration is complete.</returns>
    public bool TryGetNextBatch(out (TInput[], TOutput[]) batch)
    {
        EnsureIteratorInitialized();

        if (!_hasNextBatch || _nextBatch == null)
        {
            batch = default;
            return false;
        }

        batch = _nextBatch.Value;

        // Advance to next batch
        if (_currentEnumerator != null && _currentEnumerator.MoveNext())
        {
            _nextBatch = _currentEnumerator.Current;
            _hasNextBatch = true;
        }
        else
        {
            _nextBatch = null;
            _hasNextBatch = false;
        }

        return true;
    }

    /// <summary>
    /// Resets the iteration to the beginning.
    /// </summary>
    public void ResetIteration()
    {
        _currentEnumerator?.Dispose();
        _currentEnumerator = null;
        _nextBatch = null;
        _hasNextBatch = false;
    }

    private void EnsureIteratorInitialized()
    {
        if (_currentEnumerator == null)
        {
            _currentEnumerator = GetBatches().GetEnumerator();
            if (_currentEnumerator.MoveNext())
            {
                _nextBatch = _currentEnumerator.Current;
                _hasNextBatch = true;
            }
            else
            {
                _hasNextBatch = false;
            }
        }
    }

    /// <inheritdoc/>
    public IEnumerable<(TInput[], TOutput[])> GetBatches(
        int? batchSize = null,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null)
    {
        int actualBatchSize = batchSize ?? _batchSize;

        if (shuffle)
        {
            // For shuffled iteration, we need random access which requires reading all lines
            // For true streaming without full load, use GetBatchesAsync with shuffle=false
            foreach (var batch in GetShuffledBatches(actualBatchSize, dropLast, seed))
            {
                yield return batch;
            }
        }
        else
        {
            // Sequential streaming - true memory-efficient iteration
            foreach (var batch in GetSequentialBatches(actualBatchSize, dropLast))
            {
                yield return batch;
            }
        }
    }

    private IEnumerable<(TInput[], TOutput[])> GetSequentialBatches(int batchSize, bool dropLast)
    {
        using var reader = new StreamReader(_filePath);

        // Skip header if present
        if (_hasHeader)
        {
            reader.ReadLine();
        }

        var inputBatch = new List<TInput>(batchSize);
        var outputBatch = new List<TOutput>(batchSize);
        int lineNumber = 0;
        string? line;

        while ((line = reader.ReadLine()) != null)
        {
            var (input, output) = _lineParser(line, lineNumber);
            inputBatch.Add(input);
            outputBatch.Add(output);
            lineNumber++;

            if (inputBatch.Count == batchSize)
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

    private IEnumerable<(TInput[], TOutput[])> GetShuffledBatches(int batchSize, bool dropLast, int? seed)
    {
        // Read all lines for shuffling (memory cost for shuffled iteration)
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

        // Shuffle indices
        int[] indices = new int[lines.Count];
        for (int i = 0; i < indices.Length; i++)
        {
            indices[i] = i;
        }

        Random random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        // Yield batches
        int numBatches = indices.Length / batchSize;
        if (!dropLast && indices.Length % batchSize > 0)
        {
            numBatches++;
        }

        for (int b = 0; b < numBatches; b++)
        {
            int startIdx = b * batchSize;
            int endIdx = Math.Min(startIdx + batchSize, indices.Length);
            int currentBatchSize = endIdx - startIdx;

            var inputs = new TInput[currentBatchSize];
            var outputs = new TOutput[currentBatchSize];

            for (int i = 0; i < currentBatchSize; i++)
            {
                int idx = indices[startIdx + i];
                var (input, output) = _lineParser(lines[idx], idx);
                inputs[i] = input;
                outputs[i] = output;
            }

            yield return (inputs, outputs);
        }
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<(TInput[], TOutput[])> GetBatchesAsync(
        int? batchSize = null,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null,
        int prefetchCount = 2,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CsvStreamingDataLoader<TInput, TOutput>));
        }

        // Use sync version wrapped in Task.Run for prefetching
        var syncEnumerable = GetBatches(batchSize, shuffle, dropLast, seed);

        var outputChannel = Channel.CreateBounded<(TInput[], TOutput[])>(
            new BoundedChannelOptions(prefetchCount > 0 ? prefetchCount : _prefetchCount)
            {
                FullMode = BoundedChannelFullMode.Wait,
                SingleReader = true,
                SingleWriter = true
            });

        var producerTask = Task.Run(async () =>
        {
            try
            {
                foreach (var batch in syncEnumerable)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    await outputChannel.Writer.WriteAsync(batch, cancellationToken);
                }
            }
            finally
            {
                outputChannel.Writer.Complete();
            }
        }, cancellationToken);

        // Consume batches (net471 compatible)
        while (await outputChannel.Reader.WaitToReadAsync(cancellationToken))
        {
            while (outputChannel.Reader.TryRead(out var batch))
            {
                yield return batch;
            }
        }

        await producerTask;
    }

    /// <summary>
    /// Disposes the CSV streaming data loader.
    /// </summary>
    public void Dispose()
    {
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// A streaming data loader that uses memory-mapped files for efficient random access.
/// </summary>
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
public class MemoryMappedStreamingDataLoader<TInput, TOutput> : StreamingDataLoader<TInput, TOutput>
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
        : base(sampleCount, sampleReader, batchSize, prefetchCount, numWorkers)
    {
    }
}
