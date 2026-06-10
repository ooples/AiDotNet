using System.IO;
using System.Text;

namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// Parses the GGUF weight format (used by llama.cpp and the wider GGML ecosystem) into a
/// <see cref="GgufFile"/>: the header, the metadata key/value store, and the tensor directory.
/// </summary>
/// <remarks>
/// <para>
/// Layout: magic <c>"GGUF"</c>, a version, tensor and metadata counts, the metadata KV pairs (12 value types
/// including typed arrays), then the tensor infos (name, dimensions, ggml type, offset), then aligned tensor
/// data. This reader validates the structure and exposes metadata + tensors; reading F32/F16 tensor values is
/// supported (quantized formats are listed but not yet dequantized). All integers are little-endian per the
/// spec.
/// </para>
/// <para><b>For Beginners:</b> Hand it a <c>.gguf</c> file and it tells you the model's settings and the
/// weight tensors inside — the first step in loading a llama.cpp-style model.
/// </para>
/// </remarks>
public static class GgufReader
{
    private const uint Magic = 0x46554747; // "GGUF" little-endian

    /// <summary>
    /// Parses a GGUF file from a byte array.
    /// </summary>
    /// <param name="data">The full file contents.</param>
    /// <returns>The parsed file.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="data"/> is <c>null</c>.</exception>
    /// <exception cref="InvalidDataException">Thrown when the magic, structure, or a value type is invalid.</exception>
    public static GgufFile Read(byte[] data)
    {
        Guard.NotNull(data);
        using var stream = new MemoryStream(data, writable: false);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        if (stream.Length < 8 || reader.ReadUInt32() != Magic)
        {
            throw new InvalidDataException("Not a GGUF file (missing 'GGUF' magic).");
        }

        var version = reader.ReadUInt32();
        var tensorCount = reader.ReadUInt64();
        var metadataCount = reader.ReadUInt64();

        var metadata = new Dictionary<string, object>(StringComparer.Ordinal);
        for (ulong i = 0; i < metadataCount; i++)
        {
            var key = ReadString(reader);
            var valueType = reader.ReadUInt32();
            metadata[key] = ReadValue(reader, valueType);
        }

        var tensors = new List<GgufTensorInfo>(checked((int)tensorCount));
        for (ulong i = 0; i < tensorCount; i++)
        {
            var name = ReadString(reader);
            var dimCount = reader.ReadUInt32();
            var dims = new long[dimCount];
            for (var d = 0; d < dimCount; d++)
            {
                dims[d] = checked((long)reader.ReadUInt64());
            }

            var ggmlType = reader.ReadUInt32();
            var offset = reader.ReadUInt64();
            tensors.Add(new GgufTensorInfo(name, dims, ggmlType, offset));
        }

        var alignment = metadata.TryGetValue("general.alignment", out var a) && a is uint au ? checked((int)au) : 32;
        if (alignment <= 0)
        {
            alignment = 32;
        }

        var position = stream.Position;
        var dataStart = position % alignment == 0 ? position : position + (alignment - (position % alignment));

        return new GgufFile(data, version, alignment, dataStart, metadata, tensors);
    }

    /// <summary>
    /// Parses a GGUF file from a stream (fully read into memory).
    /// </summary>
    /// <param name="stream">The stream to read.</param>
    /// <returns>The parsed file.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="stream"/> is <c>null</c>.</exception>
    public static GgufFile Read(Stream stream)
    {
        Guard.NotNull(stream);
        using var memory = new MemoryStream();
        stream.CopyTo(memory);
        return Read(memory.ToArray());
    }

    private static string ReadString(BinaryReader reader)
    {
        var length = checked((int)reader.ReadUInt64());
        var bytes = reader.ReadBytes(length);
        return Encoding.UTF8.GetString(bytes);
    }

    private static object ReadValue(BinaryReader reader, uint valueType)
    {
        switch (valueType)
        {
            case 0: return reader.ReadByte();          // UINT8
            case 1: return reader.ReadSByte();         // INT8
            case 2: return reader.ReadUInt16();        // UINT16
            case 3: return reader.ReadInt16();         // INT16
            case 4: return reader.ReadUInt32();        // UINT32
            case 5: return reader.ReadInt32();         // INT32
            case 6: return reader.ReadSingle();        // FLOAT32
            case 7: return reader.ReadByte() != 0;     // BOOL
            case 8: return ReadString(reader);         // STRING
            case 9:                                    // ARRAY
            {
                var elementType = reader.ReadUInt32();
                var count = checked((int)reader.ReadUInt64());
                var array = new object[count];
                for (var i = 0; i < count; i++)
                {
                    array[i] = ReadValue(reader, elementType);
                }

                return array;
            }

            case 10: return reader.ReadUInt64();       // UINT64
            case 11: return reader.ReadInt64();        // INT64
            case 12: return reader.ReadDouble();       // FLOAT64
            default:
                throw new InvalidDataException($"Unknown GGUF metadata value type {valueType}.");
        }
    }
}
