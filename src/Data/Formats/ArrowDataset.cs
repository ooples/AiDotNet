using System.Text;

namespace AiDotNet.Data.Formats;

/// <summary>
/// Provides read/write access to datasets in a custom binary columnar format
/// inspired by Apache Arrow IPC.
/// </summary>
/// <remarks>
/// <para>
/// <b>Important:</b> This is NOT a native Apache Arrow implementation and is NOT compatible with
/// Arrow IPC files created by PyArrow, the Arrow C++ library, or other Arrow tools. It is a custom
/// columnar format designed for efficient column-wise access in AiDotNet ML pipelines.
/// For native Arrow interop, use the Apache.Arrow NuGet package.
/// </para>
/// <para>
/// The on-disk format:
/// - Header: [magic: "ARRW" 4 bytes] [version: 4 bytes] [numRows: 4 bytes] [numColumns: 4 bytes]
/// - Column table: for each column: [nameLen: 4][name: bytes][elementType: 4][elementsPerRow: 4][dataOffset: 8][dataLength: 8]
/// - Data section: raw values for each column (doubles, 8 bytes each)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for values.</typeparam>
public class ArrowDataset<T> : IDisposable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static readonly byte[] Magic = Encoding.ASCII.GetBytes("ARRW");

    private readonly ArrowDatasetOptions _options;
    private readonly Dictionary<string, ColumnInfo> _columns;
    private FileStream? _stream;
    private BinaryReader? _reader;
    private int _numRows;
    private bool _disposed;

    private sealed class ColumnInfo
    {
        public int ElementsPerRow { get; set; }
        public long DataOffset { get; set; }
        public long DataLength { get; set; }
    }

    /// <summary>
    /// Gets the total number of rows in the dataset.
    /// </summary>
    public int NumRows => _numRows;

    /// <summary>
    /// Gets the column names.
    /// </summary>
    public IReadOnlyCollection<string> ColumnNames => _columns.Keys;

    /// <summary>
    /// Opens an Arrow dataset file.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public ArrowDataset(ArrowDatasetOptions options)
    {
        _options = options;
        _columns = new Dictionary<string, ColumnInfo>();

        string filePath = _options.DataPath;
        if (Directory.Exists(filePath))
        {
            // Look for .arrow file in directory
            var files = Directory.GetFiles(filePath, "*.arrow");
            if (files.Length > 0)
                filePath = files[0];
            else
                throw new FileNotFoundException($"No .arrow files found in directory: {_options.DataPath}");
        }

        if (File.Exists(filePath))
        {
            _stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            _reader = new BinaryReader(_stream);
            ReadHeader();
        }
    }

    /// <summary>
    /// Reads an entire column as a flat array.
    /// </summary>
    /// <param name="columnName">Name of the column.</param>
    /// <returns>Array of T values.</returns>
    public T[] ReadColumn(string columnName)
    {
        if (_reader == null || _stream == null)
            throw new InvalidOperationException("No file is open.");

        if (!_columns.TryGetValue(columnName, out var info))
            throw new KeyNotFoundException($"Column '{columnName}' not found.");

        _stream.Position = info.DataOffset;
        int numElements = (int)(info.DataLength / 8);
        var data = new T[numElements];

        for (int i = 0; i < numElements; i++)
        {
            data[i] = NumOps.FromDouble(_reader.ReadDouble());
        }

        return data;
    }

    /// <summary>
    /// Reads a batch of rows from the specified columns.
    /// </summary>
    /// <param name="columnName">Column name to read.</param>
    /// <param name="startRow">First row (0-based).</param>
    /// <param name="numRows">Number of rows to read.</param>
    /// <returns>Array of T values for the specified rows.</returns>
    public T[] ReadColumnSlice(string columnName, int startRow, int numRows)
    {
        if (_reader == null || _stream == null)
            throw new InvalidOperationException("No file is open.");

        if (!_columns.TryGetValue(columnName, out var info))
            throw new KeyNotFoundException($"Column '{columnName}' not found.");

        int actualRows = Math.Min(numRows, _numRows - startRow);
        if (actualRows <= 0)
            throw new ArgumentOutOfRangeException(nameof(startRow));

        long rowOffset = (long)startRow * info.ElementsPerRow * 8;
        _stream.Position = info.DataOffset + rowOffset;

        int totalElements = actualRows * info.ElementsPerRow;
        var data = new T[totalElements];

        for (int i = 0; i < totalElements; i++)
        {
            data[i] = NumOps.FromDouble(_reader.ReadDouble());
        }

        return data;
    }

    /// <summary>
    /// Reads feature and label columns as tensors.
    /// </summary>
    /// <param name="startRow">First row to read.</param>
    /// <param name="numRows">Number of rows to read.</param>
    /// <returns>Tuple of (features tensor, labels tensor).</returns>
    public (Tensor<T> Features, Tensor<T> Labels) ReadBatch(int startRow, int numRows)
    {
        var features = ReadColumnSlice(_options.FeatureColumn, startRow, numRows);
        var labels = ReadColumnSlice(_options.LabelColumn, startRow, numRows);

        int featureElemsPerRow = _columns[_options.FeatureColumn].ElementsPerRow;
        int labelElemsPerRow = _columns[_options.LabelColumn].ElementsPerRow;
        int actualRows = features.Length / featureElemsPerRow;

        var featureTensor = new Tensor<T>(features, new[] { actualRows, featureElemsPerRow });
        var labelTensor = labelElemsPerRow == 1
            ? new Tensor<T>(labels, new[] { actualRows })
            : new Tensor<T>(labels, new[] { actualRows, labelElemsPerRow });

        return (featureTensor, labelTensor);
    }

    /// <summary>
    /// Writes a dataset to Arrow format.
    /// </summary>
    /// <param name="filePath">Output file path.</param>
    /// <param name="columns">Dictionary of column name to (data, elementsPerRow) pairs.</param>
    /// <param name="numRows">Total number of rows.</param>
    public static void WriteFile(string filePath, IReadOnlyDictionary<string, (T[] Data, int ElementsPerRow)> columns, int numRows)
    {
        string? dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(stream);

        // Header
        writer.Write(Magic);
        writer.Write(1); // version
        writer.Write(numRows);
        writer.Write(columns.Count);

        // Column table
        long tableStart = stream.Position;
        foreach (var kvp in columns)
        {
            byte[] nameBytes = Encoding.UTF8.GetBytes(kvp.Key);
            writer.Write(nameBytes.Length);
            writer.Write(nameBytes);
            writer.Write(0); // elementType (0 = double)
            writer.Write(kvp.Value.ElementsPerRow);
            writer.Write(0L); // placeholder offset
            writer.Write(0L); // placeholder length
        }

        // Data section
        var offsets = new List<(long Offset, long Length)>();
        foreach (var kvp in columns)
        {
            long dataStart = stream.Position;
            foreach (T value in kvp.Value.Data)
            {
                writer.Write(NumOps.ToDouble(value));
            }
            offsets.Add((dataStart, stream.Position - dataStart));
        }

        // Fill in offsets
        stream.Position = tableStart;
        int idx = 0;
        foreach (var kvp in columns)
        {
            byte[] nameBytes = Encoding.UTF8.GetBytes(kvp.Key);
            stream.Position += 4 + nameBytes.Length + 4 + 4; // skip name + type + elemsPerRow
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
            throw new InvalidDataException("Not a valid Arrow dataset file.");
        }

        int version = _reader.ReadInt32();
        if (version != 1)
            throw new InvalidDataException($"Unsupported Arrow dataset version: {version}");

        _numRows = _reader.ReadInt32();
        int numColumns = _reader.ReadInt32();

        for (int c = 0; c < numColumns; c++)
        {
            int nameLen = _reader.ReadInt32();
            string name = Encoding.UTF8.GetString(_reader.ReadBytes(nameLen));
            int elementType = _reader.ReadInt32(); // reserved
            int elementsPerRow = _reader.ReadInt32();
            long dataOffset = _reader.ReadInt64();
            long dataLength = _reader.ReadInt64();

            _columns[name] = new ColumnInfo
            {
                ElementsPerRow = elementsPerRow,
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
