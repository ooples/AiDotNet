using System.IO.MemoryMappedFiles;

namespace AiDotNet.Data.Formats;

/// <summary>
/// Zero-copy dataset access via memory-mapped files for maximum I/O throughput.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Memory-mapped files allow the operating system to map file data directly into virtual memory,
/// enabling zero-copy access to large datasets without loading them entirely into RAM.
/// The OS handles paging data in and out as needed.
/// </para>
/// <para>
/// The dataset file uses a simple binary format:
/// - First 4 bytes: number of samples (int32 little-endian)
/// - Next 4 bytes: elements per sample (int32 little-endian)
/// - Remaining bytes: raw double-precision (8-byte) values, converted to/from T at read/write time
/// </para>
/// <para><b>For Beginners:</b> Use this for very large datasets that don't fit in RAM.
/// The operating system will transparently page data in and out as needed.
/// <code>
/// // Write a dataset to disk
/// MemoryMappedDataset&lt;float&gt;.WriteDatasetFile("data.bin", myTensor);
///
/// // Read samples on demand (zero-copy)
/// using var dataset = new MemoryMappedDataset&lt;float&gt;("data.bin");
/// var sample = dataset.ReadSample(42);
/// var batch = dataset.ReadBatch(new[] { 0, 10, 20 });
/// </code>
/// </para>
/// </remarks>
public class MemoryMappedDataset<T> : IDisposable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Size in bytes of each element stored on disk (double = 8 bytes).
    /// </summary>
    private const int ElementSize = 8; // stored as double

    private const int HeaderSize = 8; // 4 bytes numSamples + 4 bytes elementsPerSample

    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _accessor;
    private readonly int _numSamples;
    private readonly int _elementsPerSample;
    private bool _disposed;

    /// <summary>
    /// Gets the total number of samples in the dataset.
    /// </summary>
    public int NumSamples => _numSamples;

    /// <summary>
    /// Gets the number of elements per sample.
    /// </summary>
    public int ElementsPerSample => _elementsPerSample;

    /// <summary>
    /// Creates a new memory-mapped dataset from a file.
    /// </summary>
    /// <param name="filePath">Path to the dataset file.</param>
    public MemoryMappedDataset(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException("Dataset file not found.", filePath);

        _mmf = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        _accessor = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

        _numSamples = _accessor.ReadInt32(0);
        _elementsPerSample = _accessor.ReadInt32(4);

        if (_numSamples <= 0)
            throw new InvalidDataException($"Invalid number of samples: {_numSamples}");
        if (_elementsPerSample <= 0)
            throw new InvalidDataException($"Invalid elements per sample: {_elementsPerSample}");

        // Validate file size
        long expectedDataSize = HeaderSize + (long)_numSamples * _elementsPerSample * ElementSize;
        long actualSize = _accessor.Capacity;
        if (actualSize < expectedDataSize)
        {
            throw new InvalidDataException(
                $"File is too small. Expected at least {expectedDataSize} bytes for {_numSamples} samples " +
                $"with {_elementsPerSample} elements each, but file is only {actualSize} bytes.");
        }
    }

    /// <summary>
    /// Reads a single sample at the specified index.
    /// </summary>
    /// <param name="index">The sample index (0-based).</param>
    /// <returns>An array containing the sample's elements.</returns>
    public T[] ReadSample(int index)
    {
        if (index < 0 || index >= _numSamples)
            throw new ArgumentOutOfRangeException(nameof(index), $"Index must be in [0, {_numSamples - 1}].");

        var data = new T[_elementsPerSample];
        long baseOffset = HeaderSize + (long)index * _elementsPerSample * ElementSize;

        for (int i = 0; i < _elementsPerSample; i++)
        {
            double value = _accessor.ReadDouble(baseOffset + (long)i * ElementSize);
            data[i] = NumOps.FromDouble(value);
        }

        return data;
    }

    /// <summary>
    /// Reads a single sample as a Tensor.
    /// </summary>
    /// <param name="index">The sample index.</param>
    /// <param name="shape">Optional shape for the tensor. If null, returns a 1D tensor.</param>
    /// <returns>A tensor containing the sample data.</returns>
    public Tensor<T> ReadSampleAsTensor(int index, int[]? shape = null)
    {
        T[] data = ReadSample(index);
        int[] tensorShape = shape ?? new[] { _elementsPerSample };

        int expectedElements = 1;
        for (int d = 0; d < tensorShape.Length; d++)
        {
            expectedElements *= tensorShape[d];
        }

        if (expectedElements != _elementsPerSample)
        {
            throw new ArgumentException(
                $"Shape produces {expectedElements} elements but sample has {_elementsPerSample} elements.",
                nameof(shape));
        }

        return new Tensor<T>(data, tensorShape);
    }

    /// <summary>
    /// Reads a batch of samples at the specified indices.
    /// </summary>
    /// <param name="indices">The sample indices to read.</param>
    /// <param name="sampleShape">Optional shape for each sample (excluding batch dimension).</param>
    /// <returns>A batched tensor with shape [batchSize, ...sampleShape].</returns>
    public Tensor<T> ReadBatch(int[] indices, int[]? sampleShape = null)
    {
        if (indices is null || indices.Length == 0)
            throw new ArgumentException("Indices cannot be null or empty.", nameof(indices));

        int[] batchShape;
        if (sampleShape is not null)
        {
            batchShape = new int[sampleShape.Length + 1];
            batchShape[0] = indices.Length;
            Array.Copy(sampleShape, 0, batchShape, 1, sampleShape.Length);
        }
        else
        {
            batchShape = new[] { indices.Length, _elementsPerSample };
        }

        var result = new Tensor<T>(batchShape);
        var resultSpan = result.Data.Span;

        for (int i = 0; i < indices.Length; i++)
        {
            T[] sample = ReadSample(indices[i]);
            int dstOffset = i * _elementsPerSample;
            sample.AsSpan().CopyTo(resultSpan.Slice(dstOffset, _elementsPerSample));
        }

        return result;
    }

    /// <summary>
    /// Creates an iterator that yields all samples in sequential order.
    /// </summary>
    /// <returns>An enumerable of sample arrays.</returns>
    public IEnumerable<T[]> EnumerateSamples()
    {
        for (int i = 0; i < _numSamples; i++)
        {
            yield return ReadSample(i);
        }
    }

    /// <summary>
    /// Writes a dataset file in the expected format.
    /// </summary>
    /// <param name="filePath">Output file path.</param>
    /// <param name="data">The data tensor with shape [numSamples, elementsPerSample].</param>
    /// <remarks>
    /// <para>
    /// Data is stored as double-precision (8-byte) values regardless of the type T.
    /// This ensures compatibility across different numeric types and platforms.
    /// </para>
    /// </remarks>
    public static void WriteDatasetFile(string filePath, Tensor<T> data)
    {
        if (data.Shape.Length < 2)
            throw new ArgumentException("Data tensor must have at least 2 dimensions [numSamples, ...].");

        int numSamples = data.Shape[0];
        int elementsPerSample = 1;
        for (int d = 1; d < data.Shape.Length; d++)
        {
            elementsPerSample *= data.Shape[d];
        }

        string? dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
        using var writer = new BinaryWriter(stream);

        // Write header
        writer.Write(numSamples);
        writer.Write(elementsPerSample);

        // Write data as doubles
        var dataSpan = data.Data.Span;
        for (int i = 0; i < dataSpan.Length; i++)
        {
            double value = NumOps.ToDouble(dataSpan[i]);
            writer.Write(value);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _accessor.Dispose();
            _mmf.Dispose();
            _disposed = true;
        }
    }
}
