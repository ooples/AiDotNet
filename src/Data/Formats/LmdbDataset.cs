using System.Text;

namespace AiDotNet.Data.Formats;

/// <summary>
/// Provides read-only access to datasets stored in a custom binary key-value format
/// inspired by LMDB (Lightning Memory-Mapped Database).
/// </summary>
/// <remarks>
/// <para>
/// <b>Important:</b> This is NOT a native LMDB implementation and is NOT compatible with
/// LMDB databases created by other tools. It is a custom binary key-value store format
/// designed for efficient sequential and random access to ML training data within AiDotNet.
/// For native LMDB interop, use the LightningDB NuGet package.
/// </para>
/// <para>
/// The on-disk format is a simple binary key-value store:
/// - Header: [magic: 4 bytes "LMDB"] [version: 4 bytes] [numEntries: 4 bytes] [indexOffset: 8 bytes]
/// - Data section: sequential key-value pairs [keyLen: 4][key: bytes][valueLen: 4][value: bytes]
/// - Index section: [offset: 8 bytes] per entry, pointing to start of each key-value pair
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for values.</typeparam>
public class LmdbDataset<T> : IDisposable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static readonly byte[] Magic = Encoding.ASCII.GetBytes("LMDB");

    private readonly LmdbDatasetOptions _options;
    private readonly Dictionary<string, byte[]> _data;
    private readonly List<string> _keys;
    private bool _disposed;

    /// <summary>
    /// Gets the number of entries in the dataset.
    /// </summary>
    public int Count => _keys.Count;

    /// <summary>
    /// Gets all keys in the dataset.
    /// </summary>
    public IReadOnlyList<string> Keys => _keys;

    /// <summary>
    /// Opens an LMDB dataset from a directory containing a data.mdb file.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public LmdbDataset(LmdbDatasetOptions options)
    {
        _options = options;
        _data = new Dictionary<string, byte[]>();
        _keys = new List<string>();

        string dbPath = Path.Combine(_options.DataPath, "data.mdb");
        if (File.Exists(dbPath))
        {
            LoadFromFile(dbPath);
        }
    }

    /// <summary>
    /// Gets a raw byte value by key.
    /// </summary>
    /// <param name="key">The key to look up.</param>
    /// <returns>The raw byte value, or null if not found.</returns>
    public byte[]? Get(string key)
    {
        return _data.TryGetValue(key, out var value) ? value : null;
    }

    /// <summary>
    /// Gets a value as a double array (assumes 8-byte doubles stored sequentially).
    /// </summary>
    /// <param name="key">The key to look up.</param>
    /// <returns>Array of T values, or null if key not found.</returns>
    public T[]? GetAsArray(string key)
    {
        var bytes = Get(key);
        if (bytes == null) return null;

        int numElements = bytes.Length / 8;
        var result = new T[numElements];
        for (int i = 0; i < numElements; i++)
        {
            double value = BitConverter.ToDouble(bytes, i * 8);
            result[i] = NumOps.FromDouble(value);
        }

        return result;
    }

    /// <summary>
    /// Gets a value as a string.
    /// </summary>
    /// <param name="key">The key to look up.</param>
    /// <returns>String value, or null if key not found.</returns>
    public string? GetAsString(string key)
    {
        var bytes = Get(key);
        return bytes != null ? Encoding.UTF8.GetString(bytes) : null;
    }

    /// <summary>
    /// Gets a value by integer index.
    /// </summary>
    /// <param name="index">0-based index into the key list.</param>
    /// <returns>The raw byte value.</returns>
    public byte[] GetByIndex(int index)
    {
        if (index < 0 || index >= _keys.Count)
            throw new ArgumentOutOfRangeException(nameof(index));
        return _data[_keys[index]];
    }

    /// <summary>
    /// Writes a dataset to LMDB format.
    /// </summary>
    /// <param name="dataPath">Directory to write the data.mdb file.</param>
    /// <param name="entries">Key-value pairs to store.</param>
    public static void WriteDataset(string dataPath, IReadOnlyList<KeyValuePair<string, byte[]>> entries)
    {
        Directory.CreateDirectory(dataPath);
        string filePath = Path.Combine(dataPath, "data.mdb");

        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(stream);

        // Write magic and version
        writer.Write(Magic);
        writer.Write(1); // version
        writer.Write(entries.Count);
        writer.Write(0L); // placeholder for index offset

        // Write data entries and record offsets
        var offsets = new long[entries.Count];
        for (int i = 0; i < entries.Count; i++)
        {
            offsets[i] = stream.Position;
            byte[] keyBytes = Encoding.UTF8.GetBytes(entries[i].Key);
            writer.Write(keyBytes.Length);
            writer.Write(keyBytes);
            writer.Write(entries[i].Value.Length);
            writer.Write(entries[i].Value);
        }

        // Write index
        long indexOffset = stream.Position;
        foreach (long offset in offsets)
        {
            writer.Write(offset);
        }

        // Update index offset in header
        stream.Position = 12; // after magic + version + count
        writer.Write(indexOffset);
    }

    private void LoadFromFile(string filePath)
    {
        using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var reader = new BinaryReader(stream);

        // Read and verify header
        byte[] magic = reader.ReadBytes(4);
        if (magic.Length < 4 || magic[0] != Magic[0] || magic[1] != Magic[1] ||
            magic[2] != Magic[2] || magic[3] != Magic[3])
        {
            throw new InvalidDataException("Not a valid LMDB dataset file.");
        }

        int version = reader.ReadInt32();
        if (version != 1)
            throw new InvalidDataException($"Unsupported LMDB dataset version: {version}");

        int numEntries = reader.ReadInt32();
        long indexOffset = reader.ReadInt64();

        // Read index to get entry offsets
        stream.Position = indexOffset;
        var offsets = new long[numEntries];
        for (int i = 0; i < numEntries; i++)
        {
            offsets[i] = reader.ReadInt64();
        }

        // Read entries
        for (int i = 0; i < numEntries; i++)
        {
            stream.Position = offsets[i];
            int keyLen = reader.ReadInt32();
            string key = Encoding.UTF8.GetString(reader.ReadBytes(keyLen));
            int valueLen = reader.ReadInt32();
            byte[] value = reader.ReadBytes(valueLen);

            _data[key] = value;
            _keys.Add(key);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _data.Clear();
            _keys.Clear();
            _disposed = true;
        }
    }
}
