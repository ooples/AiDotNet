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
public sealed class GgufFile : INamedTensorSource
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

    /// <inheritdoc/>
    public IReadOnlyCollection<string> TensorNames => _byName.Keys;

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
        var result = new double[count];

        switch (tensor.GgmlType)
        {
            case GgufTensorInfo.TypeF32:
            {
                var offset = ValidateSpan(tensor, count, byteLength: (long)count * 4, blockSize: 1);
                for (var i = 0; i < count; i++)
                {
                    result[i] = BitConverter.ToSingle(_data, offset + (i * 4));
                }

                return result;
            }

            case GgufTensorInfo.TypeF16:
            {
                var offset = ValidateSpan(tensor, count, byteLength: (long)count * 2, blockSize: 1);
                for (var i = 0; i < count; i++)
                {
                    result[i] = HalfToFloat(BitConverter.ToUInt16(_data, offset + (i * 2)));
                }

                return result;
            }

            case GgufTensorInfo.TypeQ4_0:
                DequantizeQ4_0(ValidateBlockSpan(tensor, count, blockBytes: 18, GgufTensorInfo.QuantBlockSize), count, result);
                return result;

            case GgufTensorInfo.TypeQ4_1:
                DequantizeQ4_1(ValidateBlockSpan(tensor, count, blockBytes: 20, GgufTensorInfo.QuantBlockSize), count, result);
                return result;

            case GgufTensorInfo.TypeQ8_0:
                DequantizeQ8_0(ValidateBlockSpan(tensor, count, blockBytes: 34, GgufTensorInfo.QuantBlockSize), count, result);
                return result;

            case GgufTensorInfo.TypeQ4_K:
                DequantizeQ4_K(ValidateBlockSpan(tensor, count, blockBytes: 144, GgufTensorInfo.SuperBlockSize), count, result);
                return result;

            case GgufTensorInfo.TypeQ6_K:
                DequantizeQ6_K(ValidateBlockSpan(tensor, count, blockBytes: 210, GgufTensorInfo.SuperBlockSize), count, result);
                return result;

            default:
                throw new NotSupportedException(
                    $"Tensor '{name}' uses ggml type {tensor.GgmlType}; only F32/F16/Q4_0/Q4_1/Q8_0/Q4_K/Q6_K are " +
                    "dequantized.");
        }
    }

    /// <summary>
    /// Centralized span validation: every decode path goes through here (directly or via
    /// <see cref="ValidateBlockSpan"/>) before touching <c>_data</c>, so a malformed GGUF directory can never
    /// drive a read past the payload or silently decode a partially-filled tensor.
    /// </summary>
    /// <returns>The validated absolute offset of the tensor's bytes within <c>_data</c>.</returns>
    private int ValidateSpan(GgufTensorInfo tensor, int count, long byteLength, int blockSize)
    {
        if (count % blockSize != 0)
        {
            throw new System.IO.InvalidDataException(
                $"Tensor '{tensor.Name}' element count {count} is not aligned to its {blockSize}-value quantization block size.");
        }

        var offset = checked(_dataStart + (long)tensor.Offset);
        if (offset < 0 || byteLength < 0 || offset > _data.Length - byteLength)
        {
            throw new System.IO.InvalidDataException(
                $"Tensor '{tensor.Name}' points outside the GGUF payload.");
        }

        return checked((int)offset);
    }

    /// <summary>Validates a quantized tensor's block alignment and byte span, returning its absolute offset.</summary>
    private int ValidateBlockSpan(GgufTensorInfo tensor, int count, int blockBytes, int blockSize) =>
        ValidateSpan(tensor, count, byteLength: (long)(count / blockSize) * blockBytes, blockSize: blockSize);

    // Q4_0: per 32-value block = fp16 scale + 16 bytes of packed 4-bit quants; value = (nibble - 8) * scale.
    private void DequantizeQ4_0(int offset, int count, double[] result)
    {
        const int blockBytes = 2 + 16;
        var blocks = count / GgufTensorInfo.QuantBlockSize;
        for (var b = 0; b < blocks; b++)
        {
            var p = offset + (b * blockBytes);
            var scale = HalfToFloat(BitConverter.ToUInt16(_data, p));
            var qs = p + 2;
            var outBase = b * GgufTensorInfo.QuantBlockSize;
            for (var j = 0; j < 16; j++)
            {
                var packed = _data[qs + j];
                result[outBase + j] = ((packed & 0x0F) - 8) * scale;
                result[outBase + j + 16] = ((packed >> 4) - 8) * scale;
            }
        }
    }

    // Q4_1: per block = fp16 scale + fp16 min + 16 bytes of 4-bit quants; value = nibble * scale + min.
    private void DequantizeQ4_1(int offset, int count, double[] result)
    {
        const int blockBytes = 2 + 2 + 16;
        var blocks = count / GgufTensorInfo.QuantBlockSize;
        for (var b = 0; b < blocks; b++)
        {
            var p = offset + (b * blockBytes);
            var scale = HalfToFloat(BitConverter.ToUInt16(_data, p));
            var min = HalfToFloat(BitConverter.ToUInt16(_data, p + 2));
            var qs = p + 4;
            var outBase = b * GgufTensorInfo.QuantBlockSize;
            for (var j = 0; j < 16; j++)
            {
                var packed = _data[qs + j];
                result[outBase + j] = ((packed & 0x0F) * scale) + min;
                result[outBase + j + 16] = ((packed >> 4) * scale) + min;
            }
        }
    }

    // Q8_0: per block = fp16 scale + 32 signed 8-bit quants; value = q * scale.
    private void DequantizeQ8_0(int offset, int count, double[] result)
    {
        const int blockBytes = 2 + 32;
        var blocks = count / GgufTensorInfo.QuantBlockSize;
        for (var b = 0; b < blocks; b++)
        {
            var p = offset + (b * blockBytes);
            var scale = HalfToFloat(BitConverter.ToUInt16(_data, p));
            var qs = p + 2;
            var outBase = b * GgufTensorInfo.QuantBlockSize;
            for (var j = 0; j < GgufTensorInfo.QuantBlockSize; j++)
            {
                result[outBase + j] = unchecked((sbyte)_data[qs + j]) * scale;
            }
        }
    }

    // Q4_K super-block (256 values, 144 bytes): fp16 d + fp16 dmin + 12 packed 6-bit scale/min pairs + 128
    // bytes of 4-bit quants. value = d*scale*q - dmin*min, with 8 sub-blocks of 32. Layout matches
    // llama.cpp ggml block_q4_K / dequantize_row_q4_K exactly.
    private void DequantizeQ4_K(int offset, int count, double[] result)
    {
        const int blockBytes = 2 + 2 + 12 + 128;
        var superBlocks = count / GgufTensorInfo.SuperBlockSize;
        for (var sb = 0; sb < superBlocks; sb++)
        {
            var p = offset + (sb * blockBytes);
            var d = HalfToFloat(BitConverter.ToUInt16(_data, p));
            var dmin = HalfToFloat(BitConverter.ToUInt16(_data, p + 2));
            var scalesAt = p + 4;
            var qs = p + 4 + 12;
            var outBase = sb * GgufTensorInfo.SuperBlockSize;

            // Process the 256 values in 4 chunks of 64; each chunk uses 32 quant bytes and two sub-block
            // scale/min pairs (low nibbles then high nibbles).
            var y = outBase;
            var qOffset = qs;
            for (var chunk = 0; chunk < 4; chunk++)
            {
                var iss = chunk * 2;
                GetScaleMinK4(iss, scalesAt, out var sc1, out var m1);
                GetScaleMinK4(iss + 1, scalesAt, out var sc2, out var m2);
                var d1 = d * sc1;
                var min1 = dmin * m1;
                var d2 = d * sc2;
                var min2 = dmin * m2;

                for (var l = 0; l < 32; l++)
                {
                    result[y + l] = (d1 * (_data[qOffset + l] & 0x0F)) - min1;
                }

                for (var l = 0; l < 32; l++)
                {
                    result[y + 32 + l] = (d2 * (_data[qOffset + l] >> 4)) - min2;
                }

                y += 64;
                qOffset += 32;
            }
        }
    }

    // Unpacks the 6-bit scale (d) and min (m) for sub-block j from the 12-byte packed scales array, matching
    // llama.cpp get_scale_min_k4.
    private void GetScaleMinK4(int j, int scalesAt, out int d, out int m)
    {
        if (j < 4)
        {
            d = _data[scalesAt + j] & 63;
            m = _data[scalesAt + j + 4] & 63;
        }
        else
        {
            d = (_data[scalesAt + j + 4] & 0x0F) | ((_data[scalesAt + j - 4] >> 6) << 4);
            m = (_data[scalesAt + j + 4] >> 4) | ((_data[scalesAt + j] >> 6) << 4);
        }
    }

    // Q6_K super-block (256 values, 210 bytes): 128 bytes ql (low 4 bits) + 64 bytes qh (high 2 bits) +
    // 16 signed 8-bit scales + fp16 d. value = d * scale * (q6 - 32). Layout matches llama.cpp block_q6_K /
    // dequantize_row_q6_K exactly.
    private void DequantizeQ6_K(int offset, int count, double[] result)
    {
        const int qlBytes = 128;
        const int qhBytes = 64;
        const int scaleBytes = 16;
        const int blockBytes = qlBytes + qhBytes + scaleBytes + 2;
        var superBlocks = count / GgufTensorInfo.SuperBlockSize;
        for (var sb = 0; sb < superBlocks; sb++)
        {
            var p = offset + (sb * blockBytes);
            var qlAt = p;
            var qhAt = p + qlBytes;
            var scAt = p + qlBytes + qhBytes;
            var d = HalfToFloat(BitConverter.ToUInt16(_data, p + qlBytes + qhBytes + scaleBytes));
            var outBase = sb * GgufTensorInfo.SuperBlockSize;

            // Two chunks of 128; each uses 64 ql bytes, 32 qh bytes, and 8 scale bytes.
            var y = outBase;
            var ql = qlAt;
            var qh = qhAt;
            var sc = scAt;
            for (var n = 0; n < 2; n++)
            {
                for (var l = 0; l < 32; l++)
                {
                    var iss = l / 16;
                    var qhByte = _data[qh + l];
                    var q1 = ((_data[ql + l] & 0x0F) | (((qhByte >> 0) & 3) << 4)) - 32;
                    var q2 = ((_data[ql + l + 32] & 0x0F) | (((qhByte >> 2) & 3) << 4)) - 32;
                    var q3 = ((_data[ql + l] >> 4) | (((qhByte >> 4) & 3) << 4)) - 32;
                    var q4 = ((_data[ql + l + 32] >> 4) | (((qhByte >> 6) & 3) << 4)) - 32;
                    result[y + l] = d * Signed8(_data[sc + iss]) * q1;
                    result[y + l + 32] = d * Signed8(_data[sc + iss + 2]) * q2;
                    result[y + l + 64] = d * Signed8(_data[sc + iss + 4]) * q3;
                    result[y + l + 96] = d * Signed8(_data[sc + iss + 6]) * q4;
                }

                y += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
    }

    private static int Signed8(byte value) => unchecked((sbyte)value);

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
