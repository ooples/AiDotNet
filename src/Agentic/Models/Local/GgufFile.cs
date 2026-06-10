namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// A parsed GGUF file: its version, the metadata key/value store (hyperparameters, tokenizer config, etc.),
/// the tensor directory, and access to F32/F16 tensor values. This is the structural loading primitive for
/// GGUF (the llama.cpp weight format).
/// </summary>
/// <remarks>
/// <para>
/// Metadata values are exposed as .NET objects (numbers, strings, bools, or arrays). F32 and F16 tensors can
/// be read as <see cref="double"/>; quantized tensors (Q4_K, Q8_0, …) are listed with their type/offset but
/// dequantization is a follow-up — their raw bytes are still available.
/// </para>
/// <para><b>For Beginners:</b> The opened GGUF file. Ask it for the model's settings (metadata), which weight
/// tensors it has, and read the float ones as numbers.
/// </para>
/// </remarks>
public sealed class GgufFile
{
    private readonly byte[] _data;
    private readonly long _dataStart;
    private readonly Dictionary<string, GgufTensorInfo> _byName;

    internal GgufFile(
        byte[] data,
        uint version,
        int alignment,
        long dataStart,
        IReadOnlyDictionary<string, object> metadata,
        IReadOnlyList<GgufTensorInfo> tensors)
    {
        _data = data;
        Version = version;
        Alignment = alignment;
        _dataStart = dataStart;
        Metadata = metadata;
        Tensors = tensors;
        _byName = new Dictionary<string, GgufTensorInfo>(StringComparer.Ordinal);
        foreach (var tensor in tensors)
        {
            _byName[tensor.Name] = tensor;
        }
    }

    /// <summary>Gets the GGUF format version.</summary>
    public uint Version { get; }

    /// <summary>Gets the tensor-data alignment.</summary>
    public int Alignment { get; }

    /// <summary>Gets the metadata key/value store.</summary>
    public IReadOnlyDictionary<string, object> Metadata { get; }

    /// <summary>Gets the tensor directory.</summary>
    public IReadOnlyList<GgufTensorInfo> Tensors { get; }

    /// <summary>Returns the tensor info for a name, or <c>null</c> when not present.</summary>
    /// <param name="name">The tensor name.</param>
    public GgufTensorInfo? Get(string name) => _byName.TryGetValue(name, out var tensor) ? tensor : null;

    /// <summary>Gets a metadata value, or <c>null</c> when the key is absent.</summary>
    /// <param name="key">The metadata key.</param>
    public object? GetMetadata(string key)
    {
        Guard.NotNull(key);
        return Metadata.TryGetValue(key, out var value) ? value : null;
    }

    /// <summary>
    /// Reads an F32 or F16 tensor's values as <see cref="double"/>.
    /// </summary>
    /// <param name="name">The tensor name.</param>
    /// <returns>The tensor values in stored order.</returns>
    /// <exception cref="ArgumentException">Thrown when the tensor is unknown.</exception>
    /// <exception cref="NotSupportedException">Thrown when the tensor uses a quantized type.</exception>
    public double[] ReadAsDouble(string name)
    {
        Guard.NotNull(name);
        if (!_byName.TryGetValue(name, out var tensor))
        {
            throw new ArgumentException($"Tensor '{name}' is not present in the GGUF file.", nameof(name));
        }

        var count = checked((int)tensor.ElementCount);
        var offset = checked((int)(_dataStart + (long)tensor.Offset));
        var result = new double[count];

        switch (tensor.GgmlType)
        {
            case GgufTensorInfo.TypeF32:
                for (var i = 0; i < count; i++)
                {
                    result[i] = BitConverter.ToSingle(_data, offset + (i * 4));
                }

                return result;

            case GgufTensorInfo.TypeF16:
                for (var i = 0; i < count; i++)
                {
                    result[i] = HalfToFloat(BitConverter.ToUInt16(_data, offset + (i * 2)));
                }

                return result;

            default:
                throw new NotSupportedException(
                    $"Tensor '{name}' uses quantized ggml type {tensor.GgmlType}; dequantization is not yet supported.");
        }
    }

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
