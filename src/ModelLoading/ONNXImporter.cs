using System.Runtime.InteropServices;
using AiDotNet.Interfaces;

namespace AiDotNet.ModelLoading;

/// <summary>
/// Imports weights from ONNX model files.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ONNX (Open Neural Network Exchange) is a standard format
/// for representing machine learning models.
///
/// Many pretrained models are distributed in ONNX format. This class extracts
/// the learned weights from ONNX files so you can use them in your models.
///
/// Example usage:
/// ```csharp
/// var importer = new ONNXImporter&lt;float&gt;();
///
/// // Load weights from ONNX file
/// var weights = importer.LoadWeights("model.onnx");
///
/// // Apply to your model
/// var layer = new DenseLayer&lt;float&gt;(inputSize, outputSize);
/// importer.ApplyWeights(layer, weights);
/// ```
/// </para>
/// </remarks>
public class ONNXImporter<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Whether to log import progress.
    /// </summary>
    private readonly bool _verbose;

    /// <summary>
    /// Initializes a new instance of the ONNXImporter class.
    /// </summary>
    /// <param name="verbose">Whether to log import progress.</param>
    public ONNXImporter(bool verbose = false)
    {
        _verbose = verbose;
    }

    /// <summary>
    /// Loads all initializer tensors from an ONNX file.
    /// </summary>
    /// <param name="onnxPath">Path to the .onnx file.</param>
    /// <returns>Dictionary mapping initializer names to tensors.</returns>
    /// <exception cref="ArgumentNullException">Thrown when onnxPath is null.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file doesn't exist.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the file is not a valid ONNX file.</exception>
    public Dictionary<string, Tensor<T>> LoadWeights(string onnxPath)
    {
        if (string.IsNullOrWhiteSpace(onnxPath))
            throw new ArgumentNullException(nameof(onnxPath));
        if (!File.Exists(onnxPath))
            throw new FileNotFoundException("ONNX file not found", onnxPath);

        var weights = new Dictionary<string, Tensor<T>>();

        using var stream = File.OpenRead(onnxPath);
        using var reader = new BinaryReader(stream);

        // Parse ONNX protobuf format
        // ONNX uses Protocol Buffers, which we parse manually to avoid external dependencies
        var initializers = ParseONNXInitializers(reader, stream.Length);

        foreach (var initializer in initializers)
        {
            var tensor = ConvertToTensor(initializer);
            weights[initializer.Name] = tensor;

            if (_verbose)
            {
                Console.WriteLine($"  Loaded: {initializer.Name} [{string.Join(", ", initializer.Dims)}] ({initializer.DataType})");
            }
        }

        if (_verbose)
        {
            Console.WriteLine($"Loaded {weights.Count} tensors from {onnxPath}");
        }

        return weights;
    }

    /// <summary>
    /// Gets information about tensors in an ONNX file without loading them.
    /// </summary>
    /// <param name="onnxPath">Path to the .onnx file.</param>
    /// <returns>List of tensor metadata.</returns>
    public List<ONNXTensorInfo> GetTensorInfo(string onnxPath)
    {
        if (string.IsNullOrWhiteSpace(onnxPath))
            throw new ArgumentNullException(nameof(onnxPath));
        if (!File.Exists(onnxPath))
            throw new FileNotFoundException("ONNX file not found", onnxPath);

        using var stream = File.OpenRead(onnxPath);
        using var reader = new BinaryReader(stream);

        var initializers = ParseONNXInitializers(reader, stream.Length, metadataOnly: true);

        return initializers.Select(i => new ONNXTensorInfo
        {
            Name = i.Name,
            Shape = i.Dims.Select(d => (int)d).ToArray(),
            DataType = GetDataTypeName(i.DataType),
            ElementCount = i.Dims.Aggregate(1L, (a, b) => a * b)
        }).ToList();
    }

    /// <summary>
    /// Applies loaded weights to a model using IWeightLoadable.
    /// </summary>
    /// <param name="model">The model to load weights into.</param>
    /// <param name="weights">Dictionary of weights from LoadWeights.</param>
    /// <param name="mapping">Optional name mapping function.</param>
    /// <param name="strict">If true, fails when weights can't be loaded.</param>
    /// <returns>Load result with statistics.</returns>
    public WeightLoadResult ApplyWeights(
        IWeightLoadable<T> model,
        Dictionary<string, Tensor<T>> weights,
        Func<string, string?>? mapping = null,
        bool strict = false)
    {
        return model.LoadWeights(weights, mapping, strict);
    }

    /// <summary>
    /// Parses ONNX file to extract initializer tensors.
    /// </summary>
    private List<ONNXInitializer> ParseONNXInitializers(BinaryReader reader, long fileLength, bool metadataOnly = false)
    {
        var initializers = new List<ONNXInitializer>();

        // ONNX uses protobuf format. We need to parse the wire format manually.
        // This is a simplified parser that handles the common cases.

        try
        {
            while (reader.BaseStream.Position < fileLength)
            {
                var fieldKey = ReadVarint(reader);
                if (fieldKey == 0)
                    break;

                var fieldNumber = (int)(fieldKey >> 3);
                var wireType = (int)(fieldKey & 0x7);

                switch (wireType)
                {
                    case 0: // Varint
                        ReadVarint(reader);
                        break;

                    case 1: // 64-bit
                        reader.ReadBytes(8);
                        break;

                    case 2: // Length-delimited
                        var length = (int)ReadVarint(reader);
                        if (fieldNumber == 1) // ModelProto.graph field
                        {
                            // Parse embedded graph message
                            var graphEnd = reader.BaseStream.Position + length;
                            ParseGraphForInitializers(reader, graphEnd, initializers, metadataOnly);
                        }
                        else
                        {
                            // Skip other fields
                            reader.ReadBytes(length);
                        }
                        break;

                    case 5: // 32-bit
                        reader.ReadBytes(4);
                        break;

                    default:
                        // Unknown wire type, skip rest of file
                        return initializers;
                }
            }
        }
        catch (EndOfStreamException)
        {
            // Reached end of file, return what we have
        }

        return initializers;
    }

    /// <summary>
    /// Parses the graph section for initializer tensors.
    /// </summary>
    private void ParseGraphForInitializers(BinaryReader reader, long graphEnd, List<ONNXInitializer> initializers, bool metadataOnly)
    {
        while (reader.BaseStream.Position < graphEnd)
        {
            var fieldKey = ReadVarint(reader);
            if (fieldKey == 0)
                break;

            var fieldNumber = (int)(fieldKey >> 3);
            var wireType = (int)(fieldKey & 0x7);

            if (wireType == 2) // Length-delimited
            {
                var length = (int)ReadVarint(reader);
                if (fieldNumber == 5) // GraphProto.initializer field
                {
                    var initEnd = reader.BaseStream.Position + length;
                    var initializer = ParseTensorProto(reader, initEnd, metadataOnly);
                    if (initializer != null && !string.IsNullOrEmpty(initializer.Name))
                    {
                        initializers.Add(initializer);
                    }
                }
                else
                {
                    // Skip other fields
                    reader.ReadBytes(length);
                }
            }
            else
            {
                SkipField(reader, wireType);
            }
        }
    }

    /// <summary>
    /// Parses a TensorProto message.
    /// </summary>
    private ONNXInitializer? ParseTensorProto(BinaryReader reader, long tensorEnd, bool metadataOnly)
    {
        var initializer = new ONNXInitializer();

        while (reader.BaseStream.Position < tensorEnd)
        {
            var fieldKey = ReadVarint(reader);
            if (fieldKey == 0)
                break;

            var fieldNumber = (int)(fieldKey >> 3);
            var wireType = (int)(fieldKey & 0x7);

            switch (fieldNumber)
            {
                case 1: // dims
                    if (wireType == 0)
                    {
                        initializer.Dims.Add(ReadVarint(reader));
                    }
                    else if (wireType == 2)
                    {
                        var length = (int)ReadVarint(reader);
                        var end = reader.BaseStream.Position + length;
                        while (reader.BaseStream.Position < end)
                        {
                            initializer.Dims.Add(ReadVarint(reader));
                        }
                    }
                    break;

                case 2: // data_type
                    initializer.DataType = (int)ReadVarint(reader);
                    break;

                case 3: // segment (deprecated)
                    if (wireType == 2)
                    {
                        var length = (int)ReadVarint(reader);
                        reader.ReadBytes(length);
                    }
                    break;

                case 4: // float_data (packed floats)
                    if (!metadataOnly)
                    {
                        var length = (int)ReadVarint(reader);
                        initializer.FloatData = new float[length / 4];
                        for (int i = 0; i < initializer.FloatData.Length; i++)
                        {
                            initializer.FloatData[i] = reader.ReadSingle();
                        }
                    }
                    else
                    {
                        var length = (int)ReadVarint(reader);
                        reader.ReadBytes(length);
                    }
                    break;

                case 5: // int32_data
                    if (wireType == 2)
                    {
                        var length = (int)ReadVarint(reader);
                        reader.ReadBytes(length);
                    }
                    else
                    {
                        ReadVarint(reader);
                    }
                    break;

                case 8: // name
                    var nameLength = (int)ReadVarint(reader);
                    initializer.Name = System.Text.Encoding.UTF8.GetString(reader.ReadBytes(nameLength));
                    break;

                case 9: // raw_data
                    if (!metadataOnly)
                    {
                        var length = (int)ReadVarint(reader);
                        initializer.RawData = reader.ReadBytes(length);
                    }
                    else
                    {
                        var length = (int)ReadVarint(reader);
                        reader.ReadBytes(length);
                    }
                    break;

                case 10: // double_data
                    if (wireType == 2)
                    {
                        var length = (int)ReadVarint(reader);
                        reader.ReadBytes(length);
                    }
                    break;

                default:
                    SkipField(reader, wireType);
                    break;
            }
        }

        return initializer;
    }

    /// <summary>
    /// Converts an ONNX initializer to a Tensor.
    /// </summary>
    private Tensor<T> ConvertToTensor(ONNXInitializer initializer)
    {
        var shape = initializer.Dims.Select(d => (int)d).ToArray();
        if (shape.Length == 0)
            shape = new[] { 1 };

        var tensor = new Tensor<T>(shape);

        if (initializer.RawData != null && initializer.RawData.Length > 0)
        {
            // Parse raw data based on data type
            switch (initializer.DataType)
            {
                case 1: // FLOAT
                    var floats = MemoryMarshal.Cast<byte, float>(initializer.RawData);
                    for (int i = 0; i < Math.Min(floats.Length, tensor.Length); i++)
                    {
                        tensor.Data[i] = NumOps.FromDouble(floats[i]);
                    }
                    break;

                case 11: // DOUBLE
                    var doubles = MemoryMarshal.Cast<byte, double>(initializer.RawData);
                    for (int i = 0; i < Math.Min(doubles.Length, tensor.Length); i++)
                    {
                        tensor.Data[i] = NumOps.FromDouble(doubles[i]);
                    }
                    break;

                case 10: // FLOAT16
                    var fp16 = MemoryMarshal.Cast<byte, Half>(initializer.RawData);
                    for (int i = 0; i < Math.Min(fp16.Length, tensor.Length); i++)
                    {
                        tensor.Data[i] = NumOps.FromDouble((double)fp16[i]);
                    }
                    break;

                case 16: // BFLOAT16
                    // BFloat16 is stored as uint16, need manual conversion
                    var bf16 = MemoryMarshal.Cast<byte, ushort>(initializer.RawData);
                    for (int i = 0; i < Math.Min(bf16.Length, tensor.Length); i++)
                    {
                        // BFloat16 to float: shift left by 16 bits
                        var floatBits = (uint)bf16[i] << 16;
                        // Note: BitConverter.Int32BitsToSingle not available in .NET Framework 4.7.1
                        var value = BitConverter.ToSingle(BitConverter.GetBytes((int)floatBits), 0);
                        tensor.Data[i] = NumOps.FromDouble(value);
                    }
                    break;

                default:
                    throw new NotSupportedException($"ONNX data type {initializer.DataType} not supported");
            }
        }
        else if (initializer.FloatData != null && initializer.FloatData.Length > 0)
        {
            for (int i = 0; i < Math.Min(initializer.FloatData.Length, tensor.Length); i++)
            {
                tensor.Data[i] = NumOps.FromDouble(initializer.FloatData[i]);
            }
        }

        return tensor;
    }

    /// <summary>
    /// Reads a protobuf varint.
    /// </summary>
    private static long ReadVarint(BinaryReader reader)
    {
        long result = 0;
        int shift = 0;

        while (true)
        {
            var b = reader.ReadByte();
            result |= (long)(b & 0x7F) << shift;
            if ((b & 0x80) == 0)
                break;
            shift += 7;
            if (shift > 63)
                throw new InvalidDataException("Varint too long");
        }

        return result;
    }

    /// <summary>
    /// Skips a protobuf field based on wire type.
    /// </summary>
    private static void SkipField(BinaryReader reader, int wireType)
    {
        switch (wireType)
        {
            case 0: // Varint
                ReadVarint(reader);
                break;
            case 1: // 64-bit
                reader.ReadBytes(8);
                break;
            case 2: // Length-delimited
                var length = (int)ReadVarint(reader);
                reader.ReadBytes(length);
                break;
            case 5: // 32-bit
                reader.ReadBytes(4);
                break;
        }
    }

    /// <summary>
    /// Gets the human-readable name for an ONNX data type.
    /// </summary>
    private static string GetDataTypeName(int dataType)
    {
        return dataType switch
        {
            1 => "FLOAT",
            2 => "UINT8",
            3 => "INT8",
            4 => "UINT16",
            5 => "INT16",
            6 => "INT32",
            7 => "INT64",
            8 => "STRING",
            9 => "BOOL",
            10 => "FLOAT16",
            11 => "DOUBLE",
            12 => "UINT32",
            13 => "UINT64",
            14 => "COMPLEX64",
            15 => "COMPLEX128",
            16 => "BFLOAT16",
            _ => $"UNKNOWN({dataType})"
        };
    }

    /// <summary>
    /// Internal class for parsed ONNX initializer data.
    /// </summary>
    private class ONNXInitializer
    {
        public string Name { get; set; } = string.Empty;
        public List<long> Dims { get; } = new();
        public int DataType { get; set; }
        public byte[]? RawData { get; set; }
        public float[]? FloatData { get; set; }
    }
}

/// <summary>
/// Information about a tensor in an ONNX file.
/// </summary>
public class ONNXTensorInfo
{
    /// <summary>
    /// Name of the tensor.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Shape of the tensor.
    /// </summary>
    public int[] Shape { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Data type name (e.g., "FLOAT", "FLOAT16").
    /// </summary>
    public string DataType { get; set; } = string.Empty;

    /// <summary>
    /// Total number of elements.
    /// </summary>
    public long ElementCount { get; set; }

    /// <summary>
    /// Gets a string representation.
    /// </summary>
    public override string ToString()
    {
        return $"{Name}: [{string.Join(", ", Shape)}] ({DataType})";
    }
}
