using System.Text;

namespace AiDotNet.ComputerVision.Weights;

/// <summary>
/// Reads the weights of an ONNX model (<c>ModelProto.graph.initializer</c>) without a runtime: the protobuf wire format
/// parsed directly, each <c>TensorProto</c>'s dims and data (<c>raw_data</c> or the typed repeated fields, or external
/// data beside the file) converted to float, and the graph's nodes kept so that weights an exporter left unnamed can be
/// named.
/// </summary>
/// <remarks>
/// <para>PyTorch's exporter names a <c>Linear</c>'s bias after its module but stores the weight, transposed for
/// <c>MatMul</c>, under a number. <see cref="NameMatMulWeights"/> restores the module's name: a <c>MatMul</c> whose
/// second input is an initializer, followed by an <c>Add</c> of a named <c>….bias</c> initializer, holds
/// <c>….weight</c> as <c>[in, out]</c>; it is returned as PyTorch stores it, <c>[out, in]</c>.</para>
/// </remarks>
internal sealed class OnnxWeightReader
{
    internal sealed record Node(string OpType, string[] Inputs, string[] Outputs);

    private readonly string _directory;

    private OnnxWeightReader(string path)
    {
        _directory = Path.GetDirectoryName(Path.GetFullPath(path)) ?? ".";
    }

    /// <summary>The initializers by name.</summary>
    public Dictionary<string, Tensor<float>> Initializers { get; } = new(StringComparer.Ordinal);

    /// <summary>The graph's nodes in order.</summary>
    public List<Node> Nodes { get; } = new();

    /// <summary>Reads an <c>.onnx</c> file.</summary>
    public static OnnxWeightReader Read(string path)
    {
        var reader = new OnnxWeightReader(path);
        var bytes = File.ReadAllBytes(path);
        int position = 0;
        while (position < bytes.Length)
        {
            var (field, wire) = Tag(bytes, ref position);
            if (field == 7 && wire == 2) reader.ReadGraph(Slice(bytes, ref position));
            else Skip(bytes, ref position, wire);
        }
        return reader;
    }

    // ---------------------------------------------------------------- protobuf wire format

    private static ulong Varint(byte[] b, ref int p)
    {
        ulong result = 0;
        for (int shift = 0; shift < 64; shift += 7)
        {
            byte x = b[p++];
            result |= (ulong)(x & 0x7F) << shift;
            if ((x & 0x80) == 0) return result;
        }
        throw new InvalidDataException("A protobuf varint runs past 64 bits.");
    }

    private static (int Field, int Wire) Tag(byte[] b, ref int p)
    {
        ulong tag = Varint(b, ref p);
        return ((int)(tag >> 3), (int)(tag & 7));
    }

    // A length-delimited field's bytes as (array, start, end).
    private static (byte[] Bytes, int Start, int End) Slice(byte[] b, ref int p)
    {
        int length = checked((int)Varint(b, ref p));
        int start = p;
        p += length;
        if (p > b.Length) throw new InvalidDataException("A protobuf field runs past the end of the file.");
        return (b, start, p);
    }

    private static void Skip(byte[] b, ref int p, int wire)
    {
        switch (wire)
        {
            case 0: Varint(b, ref p); break;
            case 1: p += 8; break;
            case 2: Slice(b, ref p); break;
            case 5: p += 4; break;
            default: throw new InvalidDataException($"Unsupported protobuf wire type {wire}.");
        }
    }

    private static string Text((byte[] Bytes, int Start, int End) s) => Encoding.UTF8.GetString(s.Bytes, s.Start, s.End - s.Start);

    // ---------------------------------------------------------------- ONNX messages

    private void ReadGraph((byte[] Bytes, int Start, int End) graph)
    {
        var b = graph.Bytes;
        int p = graph.Start;
        while (p < graph.End)
        {
            var (field, wire) = Tag(b, ref p);
            if (field == 1 && wire == 2) Nodes.Add(ReadNode(Slice(b, ref p)));
            else if (field == 5 && wire == 2)
            {
                var (name, tensor) = ReadTensor(Slice(b, ref p));
                Initializers[name] = tensor;
            }
            else Skip(b, ref p, wire);
        }
    }

    private static Node ReadNode((byte[] Bytes, int Start, int End) node)
    {
        var b = node.Bytes;
        int p = node.Start;
        var inputs = new List<string>();
        var outputs = new List<string>();
        string op = "";
        while (p < node.End)
        {
            var (field, wire) = Tag(b, ref p);
            if (field == 1 && wire == 2) inputs.Add(Text(Slice(b, ref p)));
            else if (field == 2 && wire == 2) outputs.Add(Text(Slice(b, ref p)));
            else if (field == 4 && wire == 2) op = Text(Slice(b, ref p));
            else Skip(b, ref p, wire);
        }
        return new Node(op, inputs.ToArray(), outputs.ToArray());
    }

    private (string Name, Tensor<float> Tensor) ReadTensor((byte[] Bytes, int Start, int End) tensor)
    {
        var b = tensor.Bytes;
        int p = tensor.Start;
        var dims = new List<long>();
        int dataType = 0;
        string name = "";
        (byte[] Bytes, int Start, int End)? raw = null;
        var floats = new List<float>();
        var doubles = new List<double>();
        var longs = new List<long>();
        var ints = new List<int>();
        var external = new Dictionary<string, string>(StringComparer.Ordinal);
        while (p < tensor.End)
        {
            var (field, wire) = Tag(b, ref p);
            switch (field)
            {
                case 1 when wire == 0: dims.Add((long)Varint(b, ref p)); break;
                case 1 when wire == 2: Packed(Slice(b, ref p), s => dims.Add((long)Varint(s.Bytes, ref s.Position))); break;
                case 2: dataType = (int)Varint(b, ref p); break;
                case 4 when wire == 2: PackedFixed(Slice(b, ref p), 4, (bytes, i) => floats.Add(BitConverter.ToSingle(bytes, i))); break;
                case 4 when wire == 5: floats.Add(BitConverter.ToSingle(b, p)); p += 4; break;
                case 5 when wire == 2: Packed(Slice(b, ref p), s => ints.Add((int)(long)Varint(s.Bytes, ref s.Position))); break;
                case 5 when wire == 0: ints.Add((int)(long)Varint(b, ref p)); break;
                case 7 when wire == 2: Packed(Slice(b, ref p), s => longs.Add((long)Varint(s.Bytes, ref s.Position))); break;
                case 7 when wire == 0: longs.Add((long)Varint(b, ref p)); break;
                case 8 when wire == 2: name = Text(Slice(b, ref p)); break;
                case 9 when wire == 2: raw = Slice(b, ref p); break;
                case 10 when wire == 2: PackedFixed(Slice(b, ref p), 8, (bytes, i) => doubles.Add(BitConverter.ToDouble(bytes, i))); break;
                case 13 when wire == 2:
                {
                    var entry = Slice(b, ref p);
                    string key = "", value = "";
                    int q = entry.Start;
                    while (q < entry.End)
                    {
                        var (f, w) = Tag(entry.Bytes, ref q);
                        if (f == 1 && w == 2) key = Text(Slice(entry.Bytes, ref q));
                        else if (f == 2 && w == 2) value = Text(Slice(entry.Bytes, ref q));
                        else Skip(entry.Bytes, ref q, w);
                    }
                    external[key] = value;
                    break;
                }
                default: Skip(b, ref p, wire); break;
            }
        }
        if (external.TryGetValue("location", out var location))
        {
            long offset = external.TryGetValue("offset", out var o) ? long.Parse(o, System.Globalization.CultureInfo.InvariantCulture) : 0;
            var file = File.ReadAllBytes(Path.Combine(_directory, location));
            long length = external.TryGetValue("length", out var l) ? long.Parse(l, System.Globalization.CultureInfo.InvariantCulture) : file.Length - offset;
            raw = (file, checked((int)offset), checked((int)(offset + length)));
        }
        long count = 1;
        foreach (long d in dims) count *= d;
        var values = new float[count];
        if (raw is { } r)
        {
            int size = dataType switch { 1 => 4, 6 => 4, 7 => 8, 10 => 2, 11 => 8, 9 => 1, 2 => 1, 3 => 1, _ => throw Unsupported(name, dataType) };
            if (r.End - r.Start != count * size) throw new InvalidDataException($"ONNX tensor '{name}' has {r.End - r.Start} bytes for {count} values.");
            for (long i = 0; i < count; i++)
            {
                int at = r.Start + (int)(i * size);
                values[i] = dataType switch
                {
                    1 => BitConverter.ToSingle(r.Bytes, at),
                    6 => BitConverter.ToInt32(r.Bytes, at),
                    7 => BitConverter.ToInt64(r.Bytes, at),
                    10 => HalfToSingle(BitConverter.ToUInt16(r.Bytes, at)),
                    11 => (float)BitConverter.ToDouble(r.Bytes, at),
                    3 => (sbyte)r.Bytes[at],
                    _ => r.Bytes[at],
                };
            }
        }
        else
        {
            IReadOnlyList<float> source = dataType switch
            {
                1 => floats,
                11 => doubles.Select(v => (float)v).ToList(),
                7 => longs.Select(v => (float)v).ToList(),
                6 or 9 or 2 or 3 or 10 => ints.Select(v => dataType == 10 ? HalfToSingle((ushort)v) : (float)v).ToList(),
                _ => throw Unsupported(name, dataType),
            };
            if (source.Count != count) throw new InvalidDataException($"ONNX tensor '{name}' has {source.Count} values for its {count}.");
            for (int i = 0; i < count; i++) values[i] = source[i];
        }
        var shape = dims.Count == 0 ? new[] { 1 } : dims.Select(d => checked((int)d)).ToArray();
        return (name, new Tensor<float>(shape, new Vector<float>(values)));
    }

    private static InvalidDataException Unsupported(string name, int dataType) =>
        new($"ONNX tensor '{name}' has data type {dataType}, which is not supported.");

    private sealed class Cursor
    {
        public byte[] Bytes = Array.Empty<byte>();
        public int Position;
    }

    private static void Packed((byte[] Bytes, int Start, int End) s, Action<Cursor> read)
    {
        var cursor = new Cursor { Bytes = s.Bytes, Position = s.Start };
        while (cursor.Position < s.End) read(cursor);
    }

    private static void PackedFixed((byte[] Bytes, int Start, int End) s, int width, Action<byte[], int> read)
    {
        for (int i = s.Start; i + width <= s.End; i += width) read(s.Bytes, i);
    }

    private static float HalfToSingle(ushort h)
    {
        int sign = (h >> 15) & 1, exponent = (h >> 10) & 0x1F, mantissa = h & 0x3FF;
        double value = exponent == 0 ? mantissa * Math.Pow(2, -24)
            : exponent == 31 ? (mantissa == 0 ? double.PositiveInfinity : double.NaN)
            : (1 + mantissa / 1024.0) * Math.Pow(2, exponent - 15);
        return (float)(sign == 1 ? -value : value);
    }

    /// <summary>
    /// The initializers with every unnamed <c>MatMul</c> weight renamed after the bias added to its product (see the
    /// class remarks), transposed to PyTorch's <c>[out, in]</c>.
    /// </summary>
    public Dictionary<string, Tensor<float>> NameMatMulWeights()
    {
        var result = new Dictionary<string, Tensor<float>>(Initializers, StringComparer.Ordinal);
        var producers = new Dictionary<string, Node>(StringComparer.Ordinal);
        foreach (var node in Nodes)
            foreach (var output in node.Outputs) producers[output] = node;
        foreach (var add in Nodes.Where(n => n.OpType == "Add" && n.Inputs.Length == 2))
        {
            string? bias = add.Inputs.FirstOrDefault(i => Initializers.ContainsKey(i) && i.EndsWith(".bias", StringComparison.Ordinal));
            string? product = add.Inputs.FirstOrDefault(i => i != bias);
            if (bias is null || product is null || !producers.TryGetValue(product, out var matmul) || matmul.OpType != "MatMul") continue;
            string weightName = matmul.Inputs[1];
            if (!Initializers.TryGetValue(weightName, out var weight) || weight.Rank != 2) continue;
            string torchName = bias.Substring(0, bias.Length - "bias".Length) + "weight";
            if (result.ContainsKey(torchName)) continue;
            int rows = weight.Shape[0], columns = weight.Shape[1];
            var transposed = new float[rows * columns];
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < columns; c++) transposed[c * rows + r] = weight[r, c];
            result[torchName] = new Tensor<float>(new[] { columns, rows }, new Vector<float>(transposed));
        }
        return result;
    }
}
