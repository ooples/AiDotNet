namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// A tensor directory entry in a GGUF file: its name, dimensions, ggml data type, and byte offset within the
/// tensor-data section.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> GGUF (the llama.cpp weight format) lists every weight array with its shape,
/// numeric type, and where its bytes start. This is one such listing.
/// </para>
/// </remarks>
public sealed class GgufTensorInfo
{
    /// <summary>The ggml type code for 32-bit float tensors.</summary>
    public const uint TypeF32 = 0;

    /// <summary>The ggml type code for 16-bit float tensors.</summary>
    public const uint TypeF16 = 1;

    /// <summary>The ggml type code for Q4_0 quantization (32-value blocks, 4-bit, single scale).</summary>
    public const uint TypeQ4_0 = 2;

    /// <summary>The ggml type code for Q4_1 quantization (32-value blocks, 4-bit, scale + min).</summary>
    public const uint TypeQ4_1 = 3;

    /// <summary>The ggml type code for Q8_0 quantization (32-value blocks, 8-bit, single scale).</summary>
    public const uint TypeQ8_0 = 8;

    /// <summary>The ggml type code for Q4_K quantization (256-value super-blocks, 4-bit, 8 sub-scales/mins).</summary>
    public const uint TypeQ4_K = 12;

    /// <summary>The ggml type code for Q6_K quantization (256-value super-blocks, 6-bit, 16 sub-scales).</summary>
    public const uint TypeQ6_K = 14;

    /// <summary>The number of values per quantization block (ggml QK).</summary>
    public const int QuantBlockSize = 32;

    /// <summary>The number of values per k-quant super-block (ggml QK_K).</summary>
    public const int SuperBlockSize = 256;

    /// <summary>
    /// Initializes a new tensor info.
    /// </summary>
    /// <param name="name">The tensor name.</param>
    /// <param name="dimensions">The tensor dimensions.</param>
    /// <param name="ggmlType">The ggml type code.</param>
    /// <param name="offset">The byte offset within the tensor-data section.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="name"/> or <paramref name="dimensions"/> is <c>null</c>.</exception>
    public GgufTensorInfo(string name, IReadOnlyList<long> dimensions, uint ggmlType, ulong offset)
    {
        Guard.NotNull(name);
        Guard.NotNull(dimensions);
        Name = name;
        Dimensions = dimensions;
        GgmlType = ggmlType;
        Offset = offset;
    }

    /// <summary>Gets the tensor name.</summary>
    public string Name { get; }

    /// <summary>Gets the tensor dimensions.</summary>
    public IReadOnlyList<long> Dimensions { get; }

    /// <summary>Gets the ggml type code (0 = F32, 1 = F16, higher = quantized formats).</summary>
    public uint GgmlType { get; }

    /// <summary>Gets the byte offset of the tensor within the data section.</summary>
    public ulong Offset { get; }

    /// <summary>Gets the total element count (product of <see cref="Dimensions"/>).</summary>
    public long ElementCount
    {
        get
        {
            long count = 1;
            foreach (var dimension in Dimensions)
            {
                count *= dimension;
            }

            return count;
        }
    }
}
