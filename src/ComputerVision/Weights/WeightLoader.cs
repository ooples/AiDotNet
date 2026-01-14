using System.Text;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Weights;

/// <summary>
/// Loads pre-trained model weights from various file formats.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class reads saved neural network weights from files.
/// Deep learning models are trained on massive datasets, and the learned parameters
/// (weights) can be saved and reloaded. This allows you to use pre-trained models
/// without training from scratch.</para>
///
/// <para>Supported formats:
/// - PyTorch (.pt, .pth) - Python pickle with tensor data
/// - SafeTensors (.safetensors) - Safe tensor format
/// - NumPy (.npy, .npz) - NumPy array format
/// - ONNX (.onnx) - Open Neural Network Exchange format
/// </para>
/// </remarks>
public class WeightLoader
{
    private readonly INumericOperations<float> _numOps;

    /// <summary>
    /// Creates a new weight loader.
    /// </summary>
    public WeightLoader()
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<float>();
    }

    /// <summary>
    /// Loads weights from a file and returns them as a dictionary.
    /// </summary>
    /// <param name="filePath">Path to the weight file.</param>
    /// <returns>Dictionary mapping layer names to weight tensors.</returns>
    public Dictionary<string, Tensor<float>> LoadWeights(string filePath)
    {
        string extension = Path.GetExtension(filePath).ToLowerInvariant();

        return extension switch
        {
            ".pt" or ".pth" => LoadPyTorchWeights(filePath),
            ".safetensors" => LoadSafeTensors(filePath),
            ".npy" => LoadNumPySingle(filePath),
            ".npz" => LoadNumPyArchive(filePath),
            ".bin" => LoadBinaryWeights(filePath),
            _ => throw new NotSupportedException($"Unsupported weight file format: {extension}")
        };
    }

    /// <summary>
    /// Loads weights from a PyTorch .pt or .pth file.
    /// </summary>
    /// <remarks>
    /// PyTorch files use Python's pickle format with a specific structure for tensors.
    /// This implementation handles the most common tensor storage formats.
    /// </remarks>
    private Dictionary<string, Tensor<float>> LoadPyTorchWeights(string filePath)
    {
        var weights = new Dictionary<string, Tensor<float>>();

        using var stream = File.OpenRead(filePath);
        using var reader = new BinaryReader(stream);

        // Check for ZIP archive (modern PyTorch format)
        byte[] header = reader.ReadBytes(4);
        stream.Position = 0;

        if (header[0] == 0x50 && header[1] == 0x4B) // "PK" - ZIP signature
        {
            return LoadPyTorchZipFormat(filePath);
        }

        // Legacy pickle format
        return LoadPyTorchPickleFormat(stream, reader);
    }

    /// <summary>
    /// Loads PyTorch weights from modern ZIP-based format.
    /// </summary>
    private Dictionary<string, Tensor<float>> LoadPyTorchZipFormat(string filePath)
    {
        var weights = new Dictionary<string, Tensor<float>>();

        using var archive = System.IO.Compression.ZipFile.OpenRead(filePath);

        // Find the data.pkl file which contains the state dict structure
        var pklEntry = archive.Entries.FirstOrDefault(e => e.Name.EndsWith(".pkl") || e.Name == "data.pkl");
        if (pklEntry == null)
        {
            throw new InvalidDataException("No pickle file found in PyTorch archive");
        }

        // Parse pickle to get tensor metadata
        using var pklStream = pklEntry.Open();
        var tensorMetadata = ParsePickleMetadata(pklStream);

        // Load tensor data from data/ folder
        foreach (var entry in archive.Entries)
        {
            if (entry.FullName.StartsWith("data/") && !entry.FullName.EndsWith("/"))
            {
                string tensorKey = Path.GetFileNameWithoutExtension(entry.Name);

                using var dataStream = entry.Open();
                using var memStream = new MemoryStream();
                dataStream.CopyTo(memStream);
                byte[] tensorBytes = memStream.ToArray();

                if (tensorMetadata.TryGetValue(tensorKey, out var meta))
                {
                    var tensor = ParseTensorData(tensorBytes, meta.Shape, meta.DType);
                    if (tensor != null && !string.IsNullOrEmpty(meta.Name))
                    {
                        weights[meta.Name] = tensor;
                    }
                }
            }
        }

        return weights;
    }

    /// <summary>
    /// Loads PyTorch weights from legacy pickle format.
    /// </summary>
    private Dictionary<string, Tensor<float>> LoadPyTorchPickleFormat(Stream stream, BinaryReader reader)
    {
        var weights = new Dictionary<string, Tensor<float>>();

        // Read pickle protocol header
        byte proto = reader.ReadByte();
        if (proto != 0x80) // PROTO opcode
        {
            stream.Position = 0;
        }
        else
        {
            byte version = reader.ReadByte();
        }

        // Parse pickle stream
        var pickleParser = new PickleParser(reader);
        var stateDict = pickleParser.Parse();

        // Convert parsed objects to tensors
        if (stateDict is Dictionary<string, object> dict)
        {
            foreach (var kvp in dict)
            {
                if (kvp.Value is TensorData tensorData)
                {
                    var tensor = CreateTensorFromData(tensorData);
                    if (tensor != null)
                    {
                        weights[kvp.Key] = tensor;
                    }
                }
            }
        }

        return weights;
    }

    /// <summary>
    /// Loads weights from SafeTensors format.
    /// </summary>
    /// <remarks>
    /// SafeTensors is a simple, safe format for storing tensors.
    /// Header is JSON with tensor metadata, followed by raw tensor data.
    /// </remarks>
    private Dictionary<string, Tensor<float>> LoadSafeTensors(string filePath)
    {
        var weights = new Dictionary<string, Tensor<float>>();

        using var stream = File.OpenRead(filePath);
        using var reader = new BinaryReader(stream);

        // Read header size (8 bytes, little endian)
        long headerSize = reader.ReadInt64();

        if (headerSize <= 0 || headerSize > 100_000_000) // Sanity check
        {
            throw new InvalidDataException("Invalid SafeTensors header size");
        }

        // Read header JSON
        byte[] headerBytes = reader.ReadBytes((int)headerSize);
        string headerJson = Encoding.UTF8.GetString(headerBytes);

        // Parse header using simple JSON parsing
        var tensorInfos = ParseSafeTensorsHeader(headerJson);

        // Read tensor data
        long dataStart = 8 + headerSize;

        foreach (var info in tensorInfos)
        {
            stream.Position = dataStart + info.DataStart;
            int byteCount = (int)(info.DataEnd - info.DataStart);
            byte[] tensorBytes = reader.ReadBytes(byteCount);

            var tensor = ParseTensorData(tensorBytes, info.Shape, info.DType);
            if (tensor != null)
            {
                weights[info.Name] = tensor;
            }
        }

        return weights;
    }

    /// <summary>
    /// Loads a single NumPy array from .npy file.
    /// </summary>
    private Dictionary<string, Tensor<float>> LoadNumPySingle(string filePath)
    {
        var weights = new Dictionary<string, Tensor<float>>();

        using var stream = File.OpenRead(filePath);
        var tensor = LoadNpyTensor(stream);

        if (tensor != null)
        {
            string name = Path.GetFileNameWithoutExtension(filePath);
            weights[name] = tensor;
        }

        return weights;
    }

    /// <summary>
    /// Loads multiple NumPy arrays from .npz archive.
    /// </summary>
    private Dictionary<string, Tensor<float>> LoadNumPyArchive(string filePath)
    {
        var weights = new Dictionary<string, Tensor<float>>();

        using var archive = System.IO.Compression.ZipFile.OpenRead(filePath);

        foreach (var entry in archive.Entries)
        {
            if (entry.Name.EndsWith(".npy"))
            {
                using var entryStream = entry.Open();
                using var memStream = new MemoryStream();
                entryStream.CopyTo(memStream);
                memStream.Position = 0;

                var tensor = LoadNpyTensor(memStream);
                if (tensor != null)
                {
                    string name = Path.GetFileNameWithoutExtension(entry.Name);
                    weights[name] = tensor;
                }
            }
        }

        return weights;
    }

    /// <summary>
    /// Loads tensor from NumPy .npy format.
    /// </summary>
    private Tensor<float>? LoadNpyTensor(Stream stream)
    {
        using var reader = new BinaryReader(stream);

        // Magic number: 0x93NUMPY
        byte[] magic = reader.ReadBytes(6);
        if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' ||
            magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y')
        {
            return null;
        }

        // Version
        byte major = reader.ReadByte();
        byte minor = reader.ReadByte();

        // Header length
        int headerLen;
        if (major == 1)
        {
            headerLen = reader.ReadUInt16();
        }
        else
        {
            headerLen = (int)reader.ReadUInt32();
        }

        // Read header (Python dict literal)
        byte[] headerBytes = reader.ReadBytes(headerLen);
        string header = Encoding.ASCII.GetString(headerBytes);

        // Parse shape and dtype from header
        var (shape, dtype, fortranOrder) = ParseNpyHeader(header);

        // Read data
        int elementCount = 1;
        foreach (int dim in shape)
        {
            elementCount *= dim;
        }

        int elementSize = GetDTypeSize(dtype);
        byte[] data = reader.ReadBytes(elementCount * elementSize);

        return ParseTensorData(data, shape, dtype);
    }

    /// <summary>
    /// Loads weights from raw binary format with shape metadata.
    /// </summary>
    private Dictionary<string, Tensor<float>> LoadBinaryWeights(string filePath)
    {
        var weights = new Dictionary<string, Tensor<float>>();

        using var stream = File.OpenRead(filePath);
        using var reader = new BinaryReader(stream);

        // Custom binary format: count, then (name_len, name, ndim, shape, dtype, data) for each tensor
        int tensorCount = reader.ReadInt32();

        for (int i = 0; i < tensorCount; i++)
        {
            // Read name
            int nameLen = reader.ReadInt32();
            string name = Encoding.UTF8.GetString(reader.ReadBytes(nameLen));

            // Read shape
            int ndim = reader.ReadInt32();
            int[] shape = new int[ndim];
            for (int d = 0; d < ndim; d++)
            {
                shape[d] = reader.ReadInt32();
            }

            // Read dtype (0=float32, 1=float64, 2=float16, 3=int32, 4=int64)
            int dtypeCode = reader.ReadInt32();
            string dtype = dtypeCode switch
            {
                0 => "float32",
                1 => "float64",
                2 => "float16",
                3 => "int32",
                4 => "int64",
                _ => "float32"
            };

            // Read data
            int elementCount = 1;
            foreach (int dim in shape)
            {
                elementCount *= dim;
            }

            int elementSize = GetDTypeSize(dtype);
            byte[] data = reader.ReadBytes(elementCount * elementSize);

            var tensor = ParseTensorData(data, shape, dtype);
            if (tensor != null)
            {
                weights[name] = tensor;
            }
        }

        return weights;
    }

    /// <summary>
    /// Parses raw tensor data bytes into a Tensor.
    /// </summary>
    private Tensor<float>? ParseTensorData(byte[] data, int[] shape, string dtype)
    {
        if (data == null || data.Length == 0 || shape == null || shape.Length == 0)
        {
            return null;
        }

        int elementCount = 1;
        foreach (int dim in shape)
        {
            elementCount *= dim;
        }

        var tensor = new Tensor<float>(shape);
        int elementSize = GetDTypeSize(dtype);

        if (data.Length < elementCount * elementSize)
        {
            return null;
        }

        for (int i = 0; i < elementCount; i++)
        {
            float value = dtype switch
            {
                "float32" or "f4" or "<f4" => BitConverter.ToSingle(data, i * 4),
                "float64" or "f8" or "<f8" => (float)BitConverter.ToDouble(data, i * 8),
                "float16" or "f2" or "<f2" => HalfToFloat(data, i * 2),
                "int32" or "i4" or "<i4" => BitConverter.ToInt32(data, i * 4),
                "int64" or "i8" or "<i8" => BitConverter.ToInt64(data, i * 8),
                "bfloat16" or "bf16" => BFloat16ToFloat(data, i * 2),
                _ => BitConverter.ToSingle(data, i * 4)
            };

            tensor[i] = value;
        }

        return tensor;
    }

    /// <summary>
    /// Converts IEEE 754 half-precision (float16) to single-precision (float32).
    /// </summary>
    private float HalfToFloat(byte[] data, int offset)
    {
        ushort bits = BitConverter.ToUInt16(data, offset);

        int sign = (bits >> 15) & 1;
        int exponent = (bits >> 10) & 0x1F;
        int mantissa = bits & 0x3FF;

        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                return sign == 0 ? 0.0f : -0.0f;
            }
            // Subnormal
            float value = mantissa / 1024.0f * (float)Math.Pow(2, -14);
            return sign == 0 ? value : -value;
        }
        else if (exponent == 31)
        {
            if (mantissa == 0)
            {
                return sign == 0 ? float.PositiveInfinity : float.NegativeInfinity;
            }
            return float.NaN;
        }

        // Normal number
        float result = (float)((1.0 + mantissa / 1024.0) * Math.Pow(2, exponent - 15));
        return sign == 0 ? result : -result;
    }

    /// <summary>
    /// Converts bfloat16 to single-precision (float32).
    /// </summary>
    private float BFloat16ToFloat(byte[] data, int offset)
    {
        // bfloat16 is just the upper 16 bits of float32
        byte[] floatBytes = new byte[4];
        floatBytes[2] = data[offset];
        floatBytes[3] = data[offset + 1];
        return BitConverter.ToSingle(floatBytes, 0);
    }

    /// <summary>
    /// Gets the byte size of a data type.
    /// </summary>
    private int GetDTypeSize(string dtype)
    {
        return dtype switch
        {
            "float32" or "f4" or "<f4" or ">f4" => 4,
            "float64" or "f8" or "<f8" or ">f8" => 8,
            "float16" or "f2" or "<f2" or ">f2" => 2,
            "bfloat16" or "bf16" => 2,
            "int32" or "i4" or "<i4" or ">i4" => 4,
            "int64" or "i8" or "<i8" or ">i8" => 8,
            "int16" or "i2" or "<i2" or ">i2" => 2,
            "int8" or "i1" => 1,
            "uint8" or "u1" => 1,
            "bool" or "b1" => 1,
            _ => 4
        };
    }

    /// <summary>
    /// Parses NumPy .npy header to extract shape and dtype.
    /// </summary>
    private (int[] shape, string dtype, bool fortranOrder) ParseNpyHeader(string header)
    {
        // Header is a Python dict literal like: {'descr': '<f4', 'fortran_order': False, 'shape': (3, 224, 224), }
        int[] shape = Array.Empty<int>();
        string dtype = "float32";
        bool fortranOrder = false;

        // Parse dtype
        int descrStart = header.IndexOf("'descr':");
        if (descrStart >= 0)
        {
            int quoteStart = header.IndexOf("'", descrStart + 8);
            int quoteEnd = header.IndexOf("'", quoteStart + 1);
            if (quoteStart >= 0 && quoteEnd > quoteStart)
            {
                dtype = header.Substring(quoteStart + 1, quoteEnd - quoteStart - 1);
            }
        }

        // Parse fortran_order
        fortranOrder = header.Contains("'fortran_order': True");

        // Parse shape
        int shapeStart = header.IndexOf("'shape':");
        if (shapeStart >= 0)
        {
            int parenStart = header.IndexOf("(", shapeStart);
            int parenEnd = header.IndexOf(")", parenStart);
            if (parenStart >= 0 && parenEnd > parenStart)
            {
                string shapeStr = header.Substring(parenStart + 1, parenEnd - parenStart - 1);
                var dims = shapeStr.Split(',')
                    .Select(s => s.Trim())
                    .Where(s => !string.IsNullOrEmpty(s))
                    .Select(int.Parse)
                    .ToArray();
                shape = dims;
            }
        }

        return (shape, dtype, fortranOrder);
    }

    /// <summary>
    /// Parses SafeTensors header JSON.
    /// </summary>
    private List<SafeTensorInfo> ParseSafeTensorsHeader(string json)
    {
        var infos = new List<SafeTensorInfo>();

        // Simple JSON parsing for SafeTensors format
        // Format: {"tensor_name": {"dtype": "F32", "shape": [3, 224, 224], "data_offsets": [0, 602112]}, ...}
        int pos = 0;

        while (pos < json.Length)
        {
            // Find tensor name
            int nameStart = json.IndexOf('"', pos);
            if (nameStart < 0) break;

            int nameEnd = json.IndexOf('"', nameStart + 1);
            if (nameEnd < 0) break;

            string name = json.Substring(nameStart + 1, nameEnd - nameStart - 1);

            // Skip __metadata__ key
            if (name == "__metadata__")
            {
                pos = json.IndexOf('}', nameEnd) + 1;
                continue;
            }

            // Find tensor info object
            int objStart = json.IndexOf('{', nameEnd);
            if (objStart < 0) break;

            int objEnd = FindMatchingBrace(json, objStart);
            if (objEnd < 0) break;

            string objStr = json.Substring(objStart, objEnd - objStart + 1);

            // Parse dtype
            string dtype = "F32";
            int dtypeStart = objStr.IndexOf("\"dtype\"");
            if (dtypeStart >= 0)
            {
                int dquoteStart = objStr.IndexOf('"', dtypeStart + 7);
                int dquoteEnd = objStr.IndexOf('"', dquoteStart + 1);
                if (dquoteStart >= 0 && dquoteEnd > dquoteStart)
                {
                    dtype = objStr.Substring(dquoteStart + 1, dquoteEnd - dquoteStart - 1);
                }
            }

            // Parse shape
            int[] shape = Array.Empty<int>();
            int shapeStart = objStr.IndexOf("\"shape\"");
            if (shapeStart >= 0)
            {
                int bracketStart = objStr.IndexOf('[', shapeStart);
                int bracketEnd = objStr.IndexOf(']', bracketStart);
                if (bracketStart >= 0 && bracketEnd > bracketStart)
                {
                    string shapeStr = objStr.Substring(bracketStart + 1, bracketEnd - bracketStart - 1);
                    if (!string.IsNullOrWhiteSpace(shapeStr))
                    {
                        shape = shapeStr.Split(',')
                            .Select(s => s.Trim())
                            .Where(s => !string.IsNullOrEmpty(s))
                            .Select(int.Parse)
                            .ToArray();
                    }
                    else
                    {
                        shape = new[] { 1 }; // Scalar
                    }
                }
            }

            // Parse data_offsets
            long dataStart = 0, dataEnd = 0;
            int offsetsStart = objStr.IndexOf("\"data_offsets\"");
            if (offsetsStart >= 0)
            {
                int bracketStart = objStr.IndexOf('[', offsetsStart);
                int bracketEnd = objStr.IndexOf(']', bracketStart);
                if (bracketStart >= 0 && bracketEnd > bracketStart)
                {
                    string offsetsStr = objStr.Substring(bracketStart + 1, bracketEnd - bracketStart - 1);
                    var offsets = offsetsStr.Split(',')
                        .Select(s => s.Trim())
                        .Where(s => !string.IsNullOrEmpty(s))
                        .Select(long.Parse)
                        .ToArray();

                    if (offsets.Length >= 2)
                    {
                        dataStart = offsets[0];
                        dataEnd = offsets[1];
                    }
                }
            }

            // Map SafeTensors dtype to standard
            string standardDtype = dtype switch
            {
                "F32" => "float32",
                "F64" => "float64",
                "F16" => "float16",
                "BF16" => "bfloat16",
                "I32" => "int32",
                "I64" => "int64",
                "I16" => "int16",
                "I8" => "int8",
                "U8" => "uint8",
                "BOOL" => "bool",
                _ => "float32"
            };

            infos.Add(new SafeTensorInfo
            {
                Name = name,
                DType = standardDtype,
                Shape = shape,
                DataStart = dataStart,
                DataEnd = dataEnd
            });

            pos = objEnd + 1;
        }

        return infos;
    }

    private int FindMatchingBrace(string json, int start)
    {
        int depth = 0;
        for (int i = start; i < json.Length; i++)
        {
            if (json[i] == '{') depth++;
            else if (json[i] == '}')
            {
                depth--;
                if (depth == 0) return i;
            }
        }
        return -1;
    }

    /// <summary>
    /// Parses pickle metadata to extract tensor information.
    /// </summary>
    private Dictionary<string, TensorMetadata> ParsePickleMetadata(Stream stream)
    {
        var metadata = new Dictionary<string, TensorMetadata>();

        using var reader = new BinaryReader(stream);
        var parser = new PickleParser(reader);
        var result = parser.Parse();

        // Extract tensor metadata from parsed pickle
        if (result is Dictionary<string, object> dict)
        {
            int index = 0;
            foreach (var kvp in dict)
            {
                if (kvp.Value is TensorData tensorData)
                {
                    metadata[index.ToString()] = new TensorMetadata
                    {
                        Name = kvp.Key,
                        Shape = tensorData.Shape,
                        DType = tensorData.DType
                    };
                    index++;
                }
            }
        }

        return metadata;
    }

    private Tensor<float>? CreateTensorFromData(TensorData data)
    {
        if (data.Data == null || data.Shape == null)
        {
            return null;
        }

        return ParseTensorData(data.Data.ToArray(), data.Shape, data.DType);
    }

    private class SafeTensorInfo
    {
        public string Name { get; set; } = string.Empty;
        public string DType { get; set; } = "float32";
        public int[] Shape { get; set; } = Array.Empty<int>();
        public long DataStart { get; set; }
        public long DataEnd { get; set; }
    }

    private class TensorMetadata
    {
        public string Name { get; set; } = string.Empty;
        public int[] Shape { get; set; } = Array.Empty<int>();
        public string DType { get; set; } = "float32";
    }
}

/// <summary>
/// Represents tensor data extracted from a weight file.
/// </summary>
public class TensorData
{
    /// <summary>
    /// The shape of the tensor.
    /// </summary>
    public int[] Shape { get; set; } = Array.Empty<int>();

    /// <summary>
    /// The data type of the tensor.
    /// </summary>
    public string DType { get; set; } = "float32";

    /// <summary>
    /// The raw byte data of the tensor.
    /// </summary>
    public byte[] Data { get; set; } = Array.Empty<byte>();
}

/// <summary>
/// Simple pickle parser for PyTorch weight files.
/// </summary>
/// <remarks>
/// This parser handles the subset of pickle protocol needed for PyTorch state dicts.
/// It supports protocols 2-5 which are commonly used by PyTorch.
/// </remarks>
internal class PickleParser
{
    private readonly BinaryReader _reader;
    private readonly Stack<object?> _stack;
    private readonly Stack<object?> _metaStack;
    private readonly Dictionary<int, object?> _memo;

    // Pickle opcodes
    private const byte PROTO = 0x80;
    private const byte FRAME = 0x95;
    private const byte STOP = 0x2E; // '.'
    private const byte MARK = 0x28; // '('
    private const byte POP = 0x30; // '0'
    private const byte POP_MARK = 0x31; // '1'
    private const byte DUP = 0x32; // '2'
    private const byte NONE = 0x4E; // 'N'
    private const byte NEWTRUE = 0x88;
    private const byte NEWFALSE = 0x89;
    private const byte BININT = 0x4A; // 'J'
    private const byte BININT1 = 0x4B; // 'K'
    private const byte BININT2 = 0x4D; // 'M'
    private const byte LONG1 = 0x8A;
    private const byte LONG4 = 0x8B;
    private const byte BINFLOAT = 0x47; // 'G'
    private const byte SHORT_BINUNICODE = 0x8C;
    private const byte BINUNICODE = 0x58; // 'X'
    private const byte BINUNICODE8 = 0x8D;
    private const byte SHORT_BINSTRING = 0x55; // 'U'
    private const byte BINSTRING = 0x54; // 'T'
    private const byte SHORT_BINBYTES = 0x8E;
    private const byte BINBYTES = 0x42; // 'B'
    private const byte BINBYTES8 = 0x8F;
    private const byte EMPTY_TUPLE = 0x29; // ')'
    private const byte TUPLE = 0x74; // 't'
    private const byte TUPLE1 = 0x85;
    private const byte TUPLE2 = 0x86;
    private const byte TUPLE3 = 0x87;
    private const byte EMPTY_LIST = 0x5D; // ']'
    private const byte LIST = 0x6C; // 'l'
    private const byte APPEND = 0x61; // 'a'
    private const byte APPENDS = 0x65; // 'e'
    private const byte EMPTY_DICT = 0x7D; // '}'
    private const byte DICT = 0x64; // 'd'
    private const byte SETITEM = 0x73; // 's'
    private const byte SETITEMS = 0x75; // 'u'
    private const byte GLOBAL = 0x63; // 'c'
    private const byte STACK_GLOBAL = 0x93;
    private const byte REDUCE = 0x52; // 'R'
    private const byte BUILD = 0x62; // 'b'
    private const byte BINPUT = 0x71; // 'q'
    private const byte LONG_BINPUT = 0x72; // 'r'
    private const byte BINGET = 0x68; // 'h'
    private const byte LONG_BINGET = 0x6A; // 'j'
    private const byte MEMOIZE = 0x94;
    private const byte NEWOBJ = 0x81;
    private const byte NEWOBJ_EX = 0x92;
    private const byte BINPERSID = 0x51; // 'Q'

    public PickleParser(BinaryReader reader)
    {
        _reader = reader;
        _stack = new Stack<object?>();
        _metaStack = new Stack<object?>();
        _memo = new Dictionary<int, object?>();
    }

    public object? Parse()
    {
        try
        {
            while (true)
            {
                if (_reader.BaseStream.Position >= _reader.BaseStream.Length)
                {
                    break;
                }

                byte opcode = _reader.ReadByte();

                switch (opcode)
                {
                    case PROTO:
                        _reader.ReadByte(); // version
                        break;

                    case FRAME:
                        _reader.ReadInt64(); // frame size
                        break;

                    case STOP:
                        return _stack.Count > 0 ? _stack.Pop() : null;

                    case MARK:
                        _metaStack.Push(_stack.Count);
                        break;

                    case POP:
                        if (_stack.Count > 0) _stack.Pop();
                        break;

                    case POP_MARK:
                        PopMark();
                        break;

                    case DUP:
                        if (_stack.Count > 0) _stack.Push(_stack.Peek());
                        break;

                    case NONE:
                        _stack.Push(null);
                        break;

                    case NEWTRUE:
                        _stack.Push(true);
                        break;

                    case NEWFALSE:
                        _stack.Push(false);
                        break;

                    case BININT:
                        _stack.Push(_reader.ReadInt32());
                        break;

                    case BININT1:
                        _stack.Push((int)_reader.ReadByte());
                        break;

                    case BININT2:
                        _stack.Push((int)_reader.ReadUInt16());
                        break;

                    case LONG1:
                        int len1 = _reader.ReadByte();
                        byte[] longBytes1 = _reader.ReadBytes(len1);
                        _stack.Push(BytesToLong(longBytes1));
                        break;

                    case LONG4:
                        int len4 = _reader.ReadInt32();
                        byte[] longBytes4 = _reader.ReadBytes(len4);
                        _stack.Push(BytesToLong(longBytes4));
                        break;

                    case BINFLOAT:
                        _stack.Push(_reader.ReadDouble());
                        break;

                    case SHORT_BINUNICODE:
                        int strLen1 = _reader.ReadByte();
                        _stack.Push(Encoding.UTF8.GetString(_reader.ReadBytes(strLen1)));
                        break;

                    case BINUNICODE:
                        int strLen4 = _reader.ReadInt32();
                        _stack.Push(Encoding.UTF8.GetString(_reader.ReadBytes(strLen4)));
                        break;

                    case BINUNICODE8:
                        long strLen8 = _reader.ReadInt64();
                        _stack.Push(Encoding.UTF8.GetString(_reader.ReadBytes((int)strLen8)));
                        break;

                    case SHORT_BINSTRING:
                        int bstrLen1 = _reader.ReadByte();
                        _stack.Push(Encoding.ASCII.GetString(_reader.ReadBytes(bstrLen1)));
                        break;

                    case BINSTRING:
                        int bstrLen4 = _reader.ReadInt32();
                        _stack.Push(Encoding.ASCII.GetString(_reader.ReadBytes(bstrLen4)));
                        break;

                    case SHORT_BINBYTES:
                        int bytesLen1 = _reader.ReadByte();
                        _stack.Push(_reader.ReadBytes(bytesLen1));
                        break;

                    case BINBYTES:
                        int bytesLen4 = _reader.ReadInt32();
                        _stack.Push(_reader.ReadBytes(bytesLen4));
                        break;

                    case BINBYTES8:
                        long bytesLen8 = _reader.ReadInt64();
                        _stack.Push(_reader.ReadBytes((int)bytesLen8));
                        break;

                    case EMPTY_TUPLE:
                        _stack.Push(Array.Empty<object>());
                        break;

                    case TUPLE:
                        _stack.Push(PopMarkToArray());
                        break;

                    case TUPLE1:
                        _stack.Push(new object?[] { _stack.Pop() });
                        break;

                    case TUPLE2:
                        var t2b = _stack.Pop();
                        var t2a = _stack.Pop();
                        _stack.Push(new object?[] { t2a, t2b });
                        break;

                    case TUPLE3:
                        var t3c = _stack.Pop();
                        var t3b = _stack.Pop();
                        var t3a = _stack.Pop();
                        _stack.Push(new object?[] { t3a, t3b, t3c });
                        break;

                    case EMPTY_LIST:
                        _stack.Push(new List<object?>());
                        break;

                    case LIST:
                        _stack.Push(new List<object?>(PopMarkToArray()));
                        break;

                    case APPEND:
                        var appendItem = _stack.Pop();
                        if (_stack.Peek() is List<object?> appendList)
                        {
                            appendList.Add(appendItem);
                        }
                        break;

                    case APPENDS:
                        var appendsItems = PopMarkToArray();
                        if (_stack.Peek() is List<object?> appendsList)
                        {
                            appendsList.AddRange(appendsItems);
                        }
                        break;

                    case EMPTY_DICT:
                        _stack.Push(new Dictionary<string, object?>());
                        break;

                    case DICT:
                        _stack.Push(PopMarkToDict());
                        break;

                    case SETITEM:
                        var siValue = _stack.Pop();
                        var siKey = _stack.Pop();
                        if (_stack.Peek() is Dictionary<string, object?> siDict && siKey != null)
                        {
                            siDict[siKey.ToString()!] = siValue;
                        }
                        break;

                    case SETITEMS:
                        var setItems = PopMarkToArray();
                        if (_stack.Peek() is Dictionary<string, object?> sisDict)
                        {
                            for (int i = 0; i < setItems.Length - 1; i += 2)
                            {
                                var key = setItems[i];
                                if (key != null)
                                {
                                    sisDict[key.ToString()!] = setItems[i + 1];
                                }
                            }
                        }
                        break;

                    case GLOBAL:
                        string module = ReadLine();
                        string name = ReadLine();
                        _stack.Push(new GlobalRef { Module = module, Name = name });
                        break;

                    case STACK_GLOBAL:
                        var sgName = _stack.Pop() as string;
                        var sgModule = _stack.Pop() as string;
                        _stack.Push(new GlobalRef { Module = sgModule ?? "", Name = sgName ?? "" });
                        break;

                    case REDUCE:
                        var reduceArgs = _stack.Pop();
                        var reduceFunc = _stack.Pop();
                        _stack.Push(HandleReduce(reduceFunc, reduceArgs));
                        break;

                    case BUILD:
                        var buildState = _stack.Pop();
                        var buildObj = _stack.Peek();
                        HandleBuild(buildObj, buildState);
                        break;

                    case BINPUT:
                        int putIdx1 = _reader.ReadByte();
                        if (_stack.Count > 0)
                        {
                            _memo[putIdx1] = _stack.Peek();
                        }
                        break;

                    case LONG_BINPUT:
                        int putIdx4 = _reader.ReadInt32();
                        if (_stack.Count > 0)
                        {
                            _memo[putIdx4] = _stack.Peek();
                        }
                        break;

                    case BINGET:
                        int getIdx1 = _reader.ReadByte();
                        _stack.Push(_memo.TryGetValue(getIdx1, out var val1) ? val1 : null);
                        break;

                    case LONG_BINGET:
                        int getIdx4 = _reader.ReadInt32();
                        _stack.Push(_memo.TryGetValue(getIdx4, out var val4) ? val4 : null);
                        break;

                    case MEMOIZE:
                        if (_stack.Count > 0)
                        {
                            _memo[_memo.Count] = _stack.Peek();
                        }
                        break;

                    case NEWOBJ:
                        var noArgs = _stack.Pop();
                        var noClass = _stack.Pop();
                        _stack.Push(HandleNewObj(noClass, noArgs));
                        break;

                    case NEWOBJ_EX:
                        var noeKwargs = _stack.Pop();
                        var noeArgs = _stack.Pop();
                        var noeClass = _stack.Pop();
                        _stack.Push(HandleNewObj(noeClass, noeArgs));
                        break;

                    case BINPERSID:
                        var pid = _stack.Pop();
                        _stack.Push(HandlePersistentId(pid));
                        break;

                    default:
                        // Unknown opcode - skip
                        break;
                }
            }
        }
        catch
        {
            // Return what we have parsed so far
        }

        return _stack.Count > 0 ? _stack.Pop() : new Dictionary<string, object?>();
    }

    private long BytesToLong(byte[] bytes)
    {
        if (bytes.Length == 0) return 0;

        long result = 0;
        for (int i = 0; i < Math.Min(bytes.Length, 8); i++)
        {
            result |= (long)bytes[i] << (i * 8);
        }

        // Handle sign extension
        if (bytes.Length > 0 && (bytes[bytes.Length - 1] & 0x80) != 0)
        {
            for (int i = bytes.Length; i < 8; i++)
            {
                result |= (long)0xFF << (i * 8);
            }
        }

        return result;
    }

    private string ReadLine()
    {
        var sb = new StringBuilder();
        while (true)
        {
            byte b = _reader.ReadByte();
            if (b == '\n') break;
            sb.Append((char)b);
        }
        return sb.ToString().TrimEnd('\r');
    }

    private void PopMark()
    {
        if (_metaStack.Count > 0)
        {
            int markPos = (int)_metaStack.Pop()!;
            while (_stack.Count > markPos)
            {
                _stack.Pop();
            }
        }
    }

    private object?[] PopMarkToArray()
    {
        if (_metaStack.Count == 0) return Array.Empty<object>();

        int markPos = (int)_metaStack.Pop()!;
        int count = _stack.Count - markPos;
        var items = new object?[count];

        for (int i = count - 1; i >= 0; i--)
        {
            items[i] = _stack.Pop();
        }

        return items;
    }

    private Dictionary<string, object?> PopMarkToDict()
    {
        var items = PopMarkToArray();
        var dict = new Dictionary<string, object?>();

        for (int i = 0; i < items.Length - 1; i += 2)
        {
            var key = items[i];
            if (key != null)
            {
                dict[key.ToString()!] = items[i + 1];
            }
        }

        return dict;
    }

    private object? HandleReduce(object? func, object? args)
    {
        if (func is GlobalRef gref)
        {
            // Handle PyTorch tensor reconstruction
            if (gref.Module == "torch._utils" && gref.Name == "_rebuild_tensor_v2")
            {
                return HandleRebuildTensor(args);
            }
            if (gref.Module == "torch" && gref.Name == "FloatStorage")
            {
                return HandleStorageRef(args, "float32");
            }
            if (gref.Module == "torch" && gref.Name == "HalfStorage")
            {
                return HandleStorageRef(args, "float16");
            }
            if (gref.Module == "torch" && gref.Name == "BFloat16Storage")
            {
                return HandleStorageRef(args, "bfloat16");
            }
            if (gref.Module == "collections" && gref.Name == "OrderedDict")
            {
                return new Dictionary<string, object?>();
            }
        }

        return new ReduceResult { Func = func, Args = args };
    }

    private object? HandleRebuildTensor(object? args)
    {
        if (args is object?[] argsArray && argsArray.Length >= 4)
        {
            var storage = argsArray[0];
            var storageOffset = argsArray[1];
            var size = argsArray[2] as object?[];
            var stride = argsArray[3];

            if (storage is StorageRef storageRef && size != null)
            {
                int[] shape = size.Select(s => Convert.ToInt32(s)).ToArray();

                return new TensorData
                {
                    Shape = shape,
                    DType = storageRef.DType,
                    Data = storageRef.Data ?? Array.Empty<byte>()
                };
            }
        }

        return null;
    }

    private object? HandleStorageRef(object? args, string dtype)
    {
        return new StorageRef { DType = dtype };
    }

    private void HandleBuild(object? obj, object? state)
    {
        if (obj is Dictionary<string, object?> dict && state is Dictionary<string, object?> stateDict)
        {
            foreach (var kvp in stateDict)
            {
                dict[kvp.Key] = kvp.Value;
            }
        }
    }

    private object? HandleNewObj(object? cls, object? args)
    {
        if (cls is GlobalRef gref)
        {
            if (gref.Module == "collections" && gref.Name == "OrderedDict")
            {
                return new Dictionary<string, object?>();
            }
        }
        return new NewObjResult { Class = cls, Args = args };
    }

    private object? HandlePersistentId(object? pid)
    {
        // Persistent IDs in PyTorch point to tensor storage
        if (pid is object?[] pidArray && pidArray.Length >= 5)
        {
            // (storage_type, key, location, size, ...)
            var storageType = pidArray[0];
            var key = pidArray[1]?.ToString();
            var size = pidArray[3];

            string dtype = "float32";
            if (storageType is GlobalRef gref)
            {
                dtype = gref.Name switch
                {
                    "HalfStorage" => "float16",
                    "FloatStorage" => "float32",
                    "DoubleStorage" => "float64",
                    "BFloat16Storage" => "bfloat16",
                    "IntStorage" => "int32",
                    "LongStorage" => "int64",
                    _ => "float32"
                };
            }

            return new StorageRef
            {
                Key = key ?? "",
                DType = dtype,
                Size = Convert.ToInt64(size ?? 0)
            };
        }

        return null;
    }

    private class GlobalRef
    {
        public string Module { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
    }

    private class ReduceResult
    {
        public object? Func { get; set; }
        public object? Args { get; set; }
    }

    private class NewObjResult
    {
        public object? Class { get; set; }
        public object? Args { get; set; }
    }

    private class StorageRef
    {
        public string Key { get; set; } = string.Empty;
        public string DType { get; set; } = "float32";
        public long Size { get; set; }
        public byte[]? Data { get; set; }
    }
}
