using System.IO;

namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// A parsed safetensors file: the tensor table of contents plus access to each tensor's raw bytes and a
/// conversion to <see cref="double"/> values. This is the format-level loading primitive; mapping the loaded
/// tensors onto a specific network's layers is architecture-specific.
/// </summary>
/// <remarks>
/// <para>
/// Supports the common float dtypes <c>F64</c>, <c>F32</c>, and <c>F16</c> for value conversion; other dtypes
/// are still listed and their raw bytes are accessible. Values are read little-endian (the safetensors
/// convention).
/// </para>
/// <para>
/// The file may be backed by an in-memory byte array or by a seekable stream
/// (see <see cref="SafetensorsReader.Read(Stream)"/>). When stream-backed, tensor bytes are read on demand —
/// the stream must remain open for the lifetime of this instance, and concurrent reads are serialized
/// internally.
/// </para>
/// <para><b>For Beginners:</b> The opened file. Ask it which tensors it has, get their shapes, and read any
/// one as an array of numbers — the raw material for loading pretrained weights into a model.
/// </para>
/// </remarks>
public sealed class SafetensorsFile : INamedTensorSource
{
    private readonly byte[]? _data;
    private readonly Stream? _stream;
    private readonly object _streamGate = new();
    private readonly long _dataStart;
    private readonly long _payloadLength;
    private readonly Dictionary<string, SafetensorsTensor> _byName;

    internal SafetensorsFile(byte[] data, long dataStart, IReadOnlyList<SafetensorsTensor> tensors)
        : this(tensors)
    {
        _data = data;
        _dataStart = dataStart;
        _payloadLength = data.Length - dataStart;
    }

    internal SafetensorsFile(Stream stream, long dataStart, IReadOnlyList<SafetensorsTensor> tensors)
        : this(tensors)
    {
        _stream = stream;
        _dataStart = dataStart;
        _payloadLength = stream.Length - dataStart;
    }

    private SafetensorsFile(IReadOnlyList<SafetensorsTensor> tensors)
    {
        Tensors = tensors;
        _byName = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        foreach (var tensor in tensors)
        {
            _byName[tensor.Name] = tensor;
        }
    }

    /// <summary>Gets the tensors in the file.</summary>
    public IReadOnlyList<SafetensorsTensor> Tensors { get; }

    /// <summary>Gets the names of all tensors.</summary>
    public IReadOnlyCollection<string> Names => _byName.Keys;

    /// <inheritdoc/>
    public IReadOnlyCollection<string> TensorNames => _byName.Keys;

    /// <summary>Returns the descriptor for a tensor, or <c>null</c> when not present.</summary>
    /// <param name="name">The tensor name.</param>
    public SafetensorsTensor? Get(string name) => _byName.TryGetValue(name, out var tensor) ? tensor : null;

    /// <summary>
    /// Copies a tensor's raw bytes.
    /// </summary>
    /// <param name="name">The tensor name.</param>
    /// <returns>The raw little-endian bytes.</returns>
    /// <exception cref="ArgumentException">Thrown when the tensor is unknown.</exception>
    /// <exception cref="InvalidDataException">Thrown when the tensor's byte range lies outside the payload.</exception>
    public byte[] GetRawBytes(string name) => ReadTensorBytes(Require(name));

    /// <summary>
    /// Reads a tensor's values as <see cref="double"/> (supports F64, F32, F16).
    /// </summary>
    /// <param name="name">The tensor name.</param>
    /// <returns>The tensor values in row-major order.</returns>
    /// <exception cref="ArgumentException">Thrown when the tensor is unknown.</exception>
    /// <exception cref="NotSupportedException">Thrown when the dtype is not a supported float type.</exception>
    /// <exception cref="InvalidDataException">
    /// Thrown when the tensor's byte range lies outside the payload or is not a whole multiple of the dtype's
    /// element size.
    /// </exception>
    public double[] ReadAsDouble(string name)
    {
        var tensor = Require(name);

        switch (tensor.DataType)
        {
            case "F64":
            {
                var bytes = ReadTensorBytes(tensor, elementSize: 8);
                var count = bytes.Length / 8;
                var result = new double[count];
                for (var i = 0; i < count; i++)
                {
                    result[i] = BitConverter.ToDouble(bytes, i * 8);
                }

                return result;
            }

            case "F32":
            {
                var bytes = ReadTensorBytes(tensor, elementSize: 4);
                var count = bytes.Length / 4;
                var result = new double[count];
                for (var i = 0; i < count; i++)
                {
                    result[i] = BitConverter.ToSingle(bytes, i * 4);
                }

                return result;
            }

            case "F16":
            {
                var bytes = ReadTensorBytes(tensor, elementSize: 2);
                var count = bytes.Length / 2;
                var result = new double[count];
                for (var i = 0; i < count; i++)
                {
                    var bits = BitConverter.ToUInt16(bytes, i * 2);
                    result[i] = HalfToFloat(bits);
                }

                return result;
            }

            default:
                throw new NotSupportedException(
                    $"Reading dtype '{tensor.DataType}' as double is not supported; use GetRawBytes for raw access.");
        }
    }

    private SafetensorsTensor Require(string name)
    {
        Guard.NotNull(name);
        if (_byName.TryGetValue(name, out var tensor))
        {
            return tensor;
        }

        throw new ArgumentException($"Tensor '{name}' is not present in the safetensors file.", nameof(name));
    }

    /// <summary>
    /// Centralized span validation + read: every raw/decode path goes through here, so no code touches the
    /// backing data with unvalidated metadata-derived offsets.
    /// </summary>
    private byte[] ReadTensorBytes(SafetensorsTensor tensor, int? elementSize = null)
    {
        // Bounds: the descriptor's byte range must lie entirely inside the
        // payload (descriptors guarantee begin >= 0 and end >= begin).
        if (tensor.EndByte > _payloadLength)
        {
            throw new InvalidDataException(
                $"Tensor '{tensor.Name}' has a byte range outside the file payload.");
        }

        // Alignment: a decode caller's element size must evenly divide the
        // byte length, or the trailing partial element would be fabricated.
        if (elementSize is { } size && tensor.ByteLength % size != 0)
        {
            throw new InvalidDataException(
                $"Tensor '{tensor.Name}' has byte length {tensor.ByteLength}, which is not a whole multiple " +
                $"of the {size}-byte '{tensor.DataType}' element size.");
        }

        var length = checked((int)tensor.ByteLength);
        var result = new byte[length];

        if (_data is { } data)
        {
            Array.Copy(data, checked((int)(_dataStart + tensor.BeginByte)), result, 0, length);
            return result;
        }

        // Stream-backed: serialize Seek+Read so concurrent tensor reads do not
        // interleave their positioning.
        if (_stream is { } stream)
        {
            lock (_streamGate)
            {
                stream.Seek(_dataStart + tensor.BeginByte, SeekOrigin.Begin);
                var read = 0;
                while (read < length)
                {
                    var n = stream.Read(result, read, length - read);
                    if (n <= 0)
                    {
                        throw new InvalidDataException(
                            $"safetensors stream ended before tensor '{tensor.Name}' was fully read.");
                    }

                    read += n;
                }
            }

            return result;
        }

        throw new InvalidOperationException("The safetensors file has no backing data source.");
    }

    // IEEE 754 half-precision (binary16) to single-precision.
    private static float HalfToFloat(ushort half)
    {
        var sign = (half >> 15) & 0x1;
        var exponent = (half >> 10) & 0x1F;
        var mantissa = half & 0x3FF;

        double value;
        if (exponent == 0)
        {
            value = mantissa * Math.Pow(2, -24);
        }
        else if (exponent == 0x1F)
        {
            value = mantissa == 0 ? double.PositiveInfinity : double.NaN;
        }
        else
        {
            value = (1 + (mantissa / 1024.0)) * Math.Pow(2, exponent - 15);
        }

        return (float)(sign == 1 ? -value : value);
    }
}
