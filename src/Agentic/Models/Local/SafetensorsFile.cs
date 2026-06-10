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
/// <para><b>For Beginners:</b> The opened file. Ask it which tensors it has, get their shapes, and read any
/// one as an array of numbers — the raw material for loading pretrained weights into a model.
/// </para>
/// </remarks>
public sealed class SafetensorsFile
{
    private readonly byte[] _data;
    private readonly long _dataStart;
    private readonly Dictionary<string, SafetensorsTensor> _byName;

    internal SafetensorsFile(byte[] data, long dataStart, IReadOnlyList<SafetensorsTensor> tensors)
    {
        _data = data;
        _dataStart = dataStart;
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

    /// <summary>Returns the descriptor for a tensor, or <c>null</c> when not present.</summary>
    /// <param name="name">The tensor name.</param>
    public SafetensorsTensor? Get(string name) => _byName.TryGetValue(name, out var tensor) ? tensor : null;

    /// <summary>
    /// Copies a tensor's raw bytes.
    /// </summary>
    /// <param name="name">The tensor name.</param>
    /// <returns>The raw little-endian bytes.</returns>
    /// <exception cref="ArgumentException">Thrown when the tensor is unknown.</exception>
    public byte[] GetRawBytes(string name)
    {
        var tensor = Require(name);
        var length = checked((int)tensor.ByteLength);
        var result = new byte[length];
        Array.Copy(_data, checked((int)(_dataStart + tensor.BeginByte)), result, 0, length);
        return result;
    }

    /// <summary>
    /// Reads a tensor's values as <see cref="double"/> (supports F64, F32, F16).
    /// </summary>
    /// <param name="name">The tensor name.</param>
    /// <returns>The tensor values in row-major order.</returns>
    /// <exception cref="ArgumentException">Thrown when the tensor is unknown.</exception>
    /// <exception cref="NotSupportedException">Thrown when the dtype is not a supported float type.</exception>
    public double[] ReadAsDouble(string name)
    {
        var tensor = Require(name);
        var offset = checked((int)(_dataStart + tensor.BeginByte));

        switch (tensor.DataType)
        {
            case "F64":
            {
                var count = checked((int)(tensor.ByteLength / 8));
                var result = new double[count];
                for (var i = 0; i < count; i++)
                {
                    result[i] = BitConverter.ToDouble(_data, offset + (i * 8));
                }

                return result;
            }

            case "F32":
            {
                var count = checked((int)(tensor.ByteLength / 4));
                var result = new double[count];
                for (var i = 0; i < count; i++)
                {
                    result[i] = BitConverter.ToSingle(_data, offset + (i * 4));
                }

                return result;
            }

            case "F16":
            {
                var count = checked((int)(tensor.ByteLength / 2));
                var result = new double[count];
                for (var i = 0; i < count; i++)
                {
                    var bits = BitConverter.ToUInt16(_data, offset + (i * 2));
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
