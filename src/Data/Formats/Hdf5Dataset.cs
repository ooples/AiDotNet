using System.Text;

namespace AiDotNet.Data.Formats;

/// <summary>
/// Provides read/write access to datasets in a custom binary format for named multidimensional arrays,
/// inspired by the HDF5 data model.
/// </summary>
/// <remarks>
/// <para>
/// <b>Important:</b> This is NOT a native HDF5 implementation and is NOT compatible with
/// HDF5 files created by h5py, HDFView, or other HDF5 tools. It is a custom binary format
/// designed for storing named tensor datasets within AiDotNet. For native HDF5 interop,
/// use the PureHDF NuGet package.
/// </para>
/// <para>
/// The on-disk format:
/// - Header: [magic: "HDF5" 4 bytes] [version: 4 bytes] [numDatasets: 4 bytes]
/// - Dataset table: for each dataset: [nameLen: 4][name: bytes][rank: 4][dims: 4*rank][dataOffset: 8][dataLength: 8]
/// - Data section: raw double-precision values for each dataset
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for values.</typeparam>
public class Hdf5Dataset<T> : IDisposable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static readonly byte[] Magic = Encoding.ASCII.GetBytes("HDF5");

    private readonly Hdf5DatasetOptions _options;
    private readonly Dictionary<string, DatasetInfo> _datasets;
    private FileStream? _stream;
    private BinaryReader? _reader;
    private bool _disposed;

    /// <summary>
    /// Information about a dataset stored in the file.
    /// </summary>
    private sealed class DatasetInfo
    {
        public int[] Shape { get; set; } = [];
        public long DataOffset { get; set; }
        public long DataLength { get; set; }
    }

    /// <summary>
    /// Gets the names of all datasets in the file.
    /// </summary>
    public IReadOnlyCollection<string> DatasetNames => _datasets.Keys;

    /// <summary>
    /// Opens an HDF5 dataset file.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public Hdf5Dataset(Hdf5DatasetOptions options)
    {
        _options = options;
        _datasets = new Dictionary<string, DatasetInfo>();

        if (File.Exists(_options.FilePath))
        {
            _stream = new FileStream(_options.FilePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            _reader = new BinaryReader(_stream);
            ReadHeader();
        }
    }

    /// <summary>
    /// Gets the shape of a named dataset.
    /// </summary>
    /// <param name="datasetName">Name of the dataset.</param>
    /// <returns>Array of dimension sizes.</returns>
    public int[] GetShape(string datasetName)
    {
        if (!_datasets.TryGetValue(datasetName, out var info))
            throw new KeyNotFoundException($"Dataset '{datasetName}' not found in file.");
        return (int[])info.Shape.Clone();
    }

    /// <summary>
    /// Reads a named dataset as a Tensor.
    /// </summary>
    /// <param name="datasetName">Name of the dataset to read.</param>
    /// <returns>Tensor containing the dataset values.</returns>
    public Tensor<T> ReadDataset(string datasetName)
    {
        if (_reader == null || _stream == null)
            throw new InvalidOperationException("No file is open.");

        if (!_datasets.TryGetValue(datasetName, out var info))
            throw new KeyNotFoundException($"Dataset '{datasetName}' not found in file.");

        _stream.Position = info.DataOffset;
        int numElements = (int)(info.DataLength / 8);
        var data = new T[numElements];

        for (int i = 0; i < numElements; i++)
        {
            double value = _reader.ReadDouble();
            data[i] = NumOps.FromDouble(value);
        }

        return new Tensor<T>(data, info.Shape);
    }

    /// <summary>
    /// Reads a slice of rows from a named dataset.
    /// </summary>
    /// <param name="datasetName">Name of the dataset.</param>
    /// <param name="startRow">First row to read (0-based).</param>
    /// <param name="numRows">Number of rows to read.</param>
    /// <returns>Tensor containing the sliced data.</returns>
    public Tensor<T> ReadSlice(string datasetName, int startRow, int numRows)
    {
        if (_reader == null || _stream == null)
            throw new InvalidOperationException("No file is open.");

        if (!_datasets.TryGetValue(datasetName, out var info))
            throw new KeyNotFoundException($"Dataset '{datasetName}' not found in file.");

        if (info.Shape.Length == 0)
            throw new InvalidOperationException("Cannot slice a scalar dataset.");

        int elementsPerRow = 1;
        for (int d = 1; d < info.Shape.Length; d++)
            elementsPerRow *= info.Shape[d];

        int maxRows = info.Shape[0];
        int actualRows = Math.Min(numRows, maxRows - startRow);
        if (actualRows <= 0)
            throw new ArgumentOutOfRangeException(nameof(startRow));

        long rowOffset = (long)startRow * elementsPerRow * 8;
        _stream.Position = info.DataOffset + rowOffset;

        int totalElements = actualRows * elementsPerRow;
        var data = new T[totalElements];
        for (int i = 0; i < totalElements; i++)
        {
            data[i] = NumOps.FromDouble(_reader.ReadDouble());
        }

        var sliceShape = new int[info.Shape.Length];
        sliceShape[0] = actualRows;
        for (int d = 1; d < info.Shape.Length; d++)
            sliceShape[d] = info.Shape[d];

        return new Tensor<T>(data, sliceShape);
    }

    /// <summary>
    /// Writes multiple named datasets to an HDF5 file.
    /// </summary>
    /// <param name="filePath">Output file path.</param>
    /// <param name="datasets">Dictionary of dataset name to (data, shape) pairs.</param>
    public static void WriteFile(string filePath, IReadOnlyDictionary<string, (T[] Data, int[] Shape)> datasets)
    {
        string? dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(stream);

        // Write magic and header
        writer.Write(Magic);
        writer.Write(1); // version
        writer.Write(datasets.Count);

        // Reserve space for dataset table (we'll fill it in after writing data)
        long tableStart = stream.Position;
        // Estimate max table size and skip past it
        foreach (var kvp in datasets)
        {
            byte[] nameBytes = Encoding.UTF8.GetBytes(kvp.Key);
            writer.Write(nameBytes.Length);
            writer.Write(nameBytes);
            writer.Write(kvp.Value.Shape.Length);
            foreach (int dim in kvp.Value.Shape)
                writer.Write(dim);
            writer.Write(0L); // placeholder data offset
            writer.Write(0L); // placeholder data length
        }

        // Write data and record offsets
        var offsets = new List<(long Offset, long Length)>();
        foreach (var kvp in datasets)
        {
            long dataStart = stream.Position;
            foreach (T value in kvp.Value.Data)
            {
                writer.Write(NumOps.ToDouble(value));
            }
            long dataLength = stream.Position - dataStart;
            offsets.Add((dataStart, dataLength));
        }

        // Go back and fill in offsets
        stream.Position = tableStart;
        int idx = 0;
        foreach (var kvp in datasets)
        {
            byte[] nameBytes = Encoding.UTF8.GetBytes(kvp.Key);
            // Skip name length + name + rank + dims
            stream.Position += 4 + nameBytes.Length + 4 + kvp.Value.Shape.Length * 4;
            writer.Write(offsets[idx].Offset);
            writer.Write(offsets[idx].Length);
            idx++;
        }
    }

    private void ReadHeader()
    {
        if (_reader == null || _stream == null) return;

        byte[] magic = _reader.ReadBytes(4);
        if (magic.Length < 4 || magic[0] != Magic[0] || magic[1] != Magic[1] ||
            magic[2] != Magic[2] || magic[3] != Magic[3])
        {
            throw new InvalidDataException("Not a valid HDF5 dataset file.");
        }

        int version = _reader.ReadInt32();
        if (version != 1)
            throw new InvalidDataException($"Unsupported HDF5 dataset version: {version}");

        int numDatasets = _reader.ReadInt32();

        for (int d = 0; d < numDatasets; d++)
        {
            int nameLen = _reader.ReadInt32();
            string name = Encoding.UTF8.GetString(_reader.ReadBytes(nameLen));
            int rank = _reader.ReadInt32();
            var shape = new int[rank];
            for (int r = 0; r < rank; r++)
                shape[r] = _reader.ReadInt32();
            long dataOffset = _reader.ReadInt64();
            long dataLength = _reader.ReadInt64();

            _datasets[name] = new DatasetInfo
            {
                Shape = shape,
                DataOffset = dataOffset,
                DataLength = dataLength
            };
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _reader?.Dispose();
            _stream?.Dispose();
            _disposed = true;
        }
    }
}
