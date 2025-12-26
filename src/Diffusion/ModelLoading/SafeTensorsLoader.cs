using System.Runtime.InteropServices;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using AiDotNet.Interfaces;

namespace AiDotNet.Diffusion.ModelLoading;

/// <summary>
/// Loads model weights from SafeTensors format files.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SafeTensors is a format developed by Hugging Face for storing model tensors safely.
/// It's the standard format for Stable Diffusion and other modern ML models.
/// </para>
/// <para>
/// <b>For Beginners:</b> SafeTensors is like a special container for AI model weights.
///
/// Why SafeTensors instead of pickle files?
/// - Safe: Cannot execute arbitrary code (unlike pickle)
/// - Fast: Memory-mapped loading for quick access
/// - Simple: Just tensors and their metadata
///
/// This loader reads SafeTensors files and converts them to our Tensor format
/// so we can use pretrained weights from HuggingFace and other sources.
///
/// File structure:
/// ```
/// [8 bytes: header length]
/// [JSON header: tensor metadata]
/// [tensor data: raw bytes]
/// ```
/// </para>
/// </remarks>
public class SafeTensorsLoader<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Loads all tensors from a SafeTensors file.
    /// </summary>
    /// <param name="path">Path to the .safetensors file.</param>
    /// <returns>Dictionary mapping tensor names to loaded tensors.</returns>
    /// <exception cref="FileNotFoundException">Thrown when the file doesn't exist.</exception>
    /// <exception cref="InvalidDataException">Thrown when the file format is invalid.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This loads all the weights from a SafeTensors file.
    ///
    /// Example usage:
    /// ```csharp
    /// var loader = new SafeTensorsLoader&lt;float&gt;();
    /// var weights = loader.Load("model.safetensors");
    /// var vaeWeight = weights["first_stage_model.encoder.conv_in.weight"];
    /// ```
    /// </para>
    /// </remarks>
    public Dictionary<string, Tensor<T>> Load(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentNullException(nameof(path));
        if (!File.Exists(path))
            throw new FileNotFoundException($"SafeTensors file not found: {path}", path);

        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        // Read header length (8 bytes, little-endian)
        var headerLength = reader.ReadInt64();
        if (headerLength <= 0 || headerLength > 100_000_000) // Sanity check: max 100MB header
        {
            throw new InvalidDataException($"Invalid header length: {headerLength}");
        }

        // Read JSON header
        var headerBytes = reader.ReadBytes((int)headerLength);
        var headerJson = System.Text.Encoding.UTF8.GetString(headerBytes);

        // Parse header as JSON
        var header = JObject.Parse(headerJson);
        if (header == null)
        {
            throw new InvalidDataException("Failed to parse SafeTensors header.");
        }

        var tensors = new Dictionary<string, Tensor<T>>();
        var headerDataOffset = 8 + headerLength; // Skip header length + header itself

        foreach (var prop in header.Properties())
        {
            var name = prop.Name;

            // Skip metadata entry
            if (name == "__metadata__")
                continue;

            var meta = prop.Value;
            if (meta == null)
                continue;

            var offsetsArray = meta["data_offsets"];
            var dtypeToken = meta["dtype"];
            var shapeArray = meta["shape"];

            if (offsetsArray == null || dtypeToken == null || shapeArray == null)
                continue;

            var offsets = offsetsArray.ToObject<long[]>();
            var dtype = dtypeToken.ToString();
            var shape = shapeArray.ToObject<int[]>();

            if (offsets == null || offsets.Length < 2 || shape == null)
                continue;

            var dataStart = offsets[0];
            var dataEnd = offsets[1];
            var dataLength = (int)(dataEnd - dataStart);

            // Seek to tensor data
            stream.Seek(headerDataOffset + dataStart, SeekOrigin.Begin);
            var data = reader.ReadBytes(dataLength);

            // Convert to Tensor<T>
            var tensor = CreateTensor(data, dtype, shape);
            tensors[name] = tensor;
        }

        return tensors;
    }

    /// <summary>
    /// Loads specific tensors from a SafeTensors file.
    /// </summary>
    /// <param name="path">Path to the .safetensors file.</param>
    /// <param name="tensorNames">Names of tensors to load.</param>
    /// <returns>Dictionary mapping tensor names to loaded tensors.</returns>
    public Dictionary<string, Tensor<T>> Load(string path, IEnumerable<string> tensorNames)
    {
        var requestedNames = new HashSet<string>(tensorNames);
        var allTensors = Load(path);

        return allTensors
            .Where(kvp => requestedNames.Contains(kvp.Key))
            .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
    }

    /// <summary>
    /// Gets the list of tensor names in a SafeTensors file without loading data.
    /// </summary>
    /// <param name="path">Path to the .safetensors file.</param>
    /// <returns>List of tensor names and their metadata.</returns>
    public List<TensorMetadata> GetTensorInfo(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentNullException(nameof(path));
        if (!File.Exists(path))
            throw new FileNotFoundException($"SafeTensors file not found: {path}", path);

        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        var headerLength = reader.ReadInt64();
        var headerBytes = reader.ReadBytes((int)headerLength);
        var headerJson = System.Text.Encoding.UTF8.GetString(headerBytes);
        var header = JObject.Parse(headerJson);

        var result = new List<TensorMetadata>();

        if (header == null)
            return result;

        foreach (var prop in header.Properties())
        {
            if (prop.Name == "__metadata__")
                continue;

            var meta = prop.Value;
            if (meta == null)
                continue;

            var dtypeToken = meta["dtype"];
            var shapeArray = meta["shape"];
            var offsetsArray = meta["data_offsets"];

            if (dtypeToken == null || shapeArray == null)
                continue;

            var shape = shapeArray.ToObject<int[]>() ?? Array.Empty<int>();
            var offsets = offsetsArray?.ToObject<long[]>();
            var dataSize = (offsets != null && offsets.Length >= 2)
                ? offsets[1] - offsets[0]
                : 0;

            result.Add(new TensorMetadata
            {
                Name = prop.Name,
                DataType = dtypeToken.ToString(),
                Shape = shape,
                DataSizeBytes = dataSize
            });
        }

        return result;
    }

    /// <summary>
    /// Creates a Tensor from raw bytes and metadata.
    /// </summary>
    private Tensor<T> CreateTensor(byte[] data, string dtype, int[] shape)
    {
        var tensor = new Tensor<T>(shape);
        var span = tensor.AsWritableSpan();

        switch (dtype.ToUpperInvariant())
        {
            case "F16":
            case "FLOAT16":
                LoadFloat16(data, span);
                break;

            case "F32":
            case "FLOAT32":
                LoadFloat32(data, span);
                break;

            case "BF16":
            case "BFLOAT16":
                LoadBFloat16(data, span);
                break;

            case "F64":
            case "FLOAT64":
                LoadFloat64(data, span);
                break;

            case "I8":
            case "INT8":
                LoadInt8(data, span);
                break;

            case "I16":
            case "INT16":
                LoadInt16(data, span);
                break;

            case "I32":
            case "INT32":
                LoadInt32(data, span);
                break;

            case "I64":
            case "INT64":
                LoadInt64(data, span);
                break;

            default:
                throw new NotSupportedException($"Unsupported dtype: {dtype}");
        }

        return tensor;
    }

    private void LoadFloat16(byte[] data, Span<T> destination)
    {
        var halfSpan = MemoryMarshal.Cast<byte, Half>(data);
        for (int i = 0; i < Math.Min(halfSpan.Length, destination.Length); i++)
        {
            destination[i] = NumOps.FromDouble((double)halfSpan[i]);
        }
    }

    private void LoadFloat32(byte[] data, Span<T> destination)
    {
        var floatSpan = MemoryMarshal.Cast<byte, float>(data);
        for (int i = 0; i < Math.Min(floatSpan.Length, destination.Length); i++)
        {
            destination[i] = NumOps.FromDouble(floatSpan[i]);
        }
    }

    private void LoadBFloat16(byte[] data, Span<T> destination)
    {
        // BFloat16 is the upper 16 bits of float32
        // Convert by padding with zeros in lower 16 bits
        for (int i = 0; i < Math.Min(data.Length / 2, destination.Length); i++)
        {
            // Read 2 bytes as bfloat16
            ushort bf16 = (ushort)(data[i * 2] | (data[i * 2 + 1] << 8));
            // Convert to float32 by shifting left 16 bits
            uint f32Bits = (uint)bf16 << 16;
            // Use BitConverter.GetBytes + ToSingle for net471 compatibility
            byte[] bytes = BitConverter.GetBytes((int)f32Bits);
            float value = BitConverter.ToSingle(bytes, 0);
            destination[i] = NumOps.FromDouble(value);
        }
    }

    private void LoadFloat64(byte[] data, Span<T> destination)
    {
        var doubleSpan = MemoryMarshal.Cast<byte, double>(data);
        for (int i = 0; i < Math.Min(doubleSpan.Length, destination.Length); i++)
        {
            destination[i] = NumOps.FromDouble(doubleSpan[i]);
        }
    }

    private void LoadInt8(byte[] data, Span<T> destination)
    {
        for (int i = 0; i < Math.Min(data.Length, destination.Length); i++)
        {
            destination[i] = NumOps.FromDouble((sbyte)data[i]);
        }
    }

    private void LoadInt16(byte[] data, Span<T> destination)
    {
        var shortSpan = MemoryMarshal.Cast<byte, short>(data);
        for (int i = 0; i < Math.Min(shortSpan.Length, destination.Length); i++)
        {
            destination[i] = NumOps.FromDouble(shortSpan[i]);
        }
    }

    private void LoadInt32(byte[] data, Span<T> destination)
    {
        var intSpan = MemoryMarshal.Cast<byte, int>(data);
        for (int i = 0; i < Math.Min(intSpan.Length, destination.Length); i++)
        {
            destination[i] = NumOps.FromDouble(intSpan[i]);
        }
    }

    private void LoadInt64(byte[] data, Span<T> destination)
    {
        var longSpan = MemoryMarshal.Cast<byte, long>(data);
        for (int i = 0; i < Math.Min(longSpan.Length, destination.Length); i++)
        {
            destination[i] = NumOps.FromDouble(longSpan[i]);
        }
    }
}

/// <summary>
/// Metadata about a tensor in a SafeTensors file.
/// </summary>
public class TensorMetadata
{
    /// <summary>
    /// Name of the tensor.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Data type (e.g., "F16", "F32", "BF16").
    /// </summary>
    public string DataType { get; set; } = string.Empty;

    /// <summary>
    /// Shape of the tensor.
    /// </summary>
    public int[] Shape { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Size of the tensor data in bytes.
    /// </summary>
    public long DataSizeBytes { get; set; }

    /// <summary>
    /// Gets the total number of elements in the tensor.
    /// </summary>
    public long ElementCount
    {
        get
        {
            if (Shape == null || Shape.Length == 0)
                return 0;
            long count = 1;
            foreach (var dim in Shape)
            {
                count *= dim;
            }
            return count;
        }
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var shapeStr = Shape != null ? string.Join(", ", Shape) : "";
        return $"{Name}: {DataType}[{shapeStr}] ({DataSizeBytes:N0} bytes)";
    }
}
