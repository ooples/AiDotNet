namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// Metadata for one tensor inside a safetensors file: its name, dtype, shape, and byte range within the data
/// section.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A safetensors file stores many named weight arrays back to back. This is the
/// "table of contents" entry for one of them — what it's called, its numeric type, its dimensions, and where
/// its bytes live in the file.
/// </para>
/// </remarks>
public sealed class SafetensorsTensor
{
    /// <summary>
    /// Initializes a new tensor descriptor.
    /// </summary>
    /// <param name="name">The tensor name.</param>
    /// <param name="dataType">The safetensors dtype string (e.g., <c>F32</c>, <c>F16</c>, <c>F64</c>).</param>
    /// <param name="shape">The tensor shape.</param>
    /// <param name="beginByte">The start offset within the data section (inclusive).</param>
    /// <param name="endByte">The end offset within the data section (exclusive).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="name"/>, <paramref name="dataType"/>, or <paramref name="shape"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="beginByte"/> is negative, <paramref name="endByte"/> is less than
    /// <paramref name="beginByte"/>, or any dimension in <paramref name="shape"/> is negative.
    /// </exception>
    public SafetensorsTensor(string name, string dataType, IReadOnlyList<long> shape, long beginByte, long endByte)
    {
        Guard.NotNull(name);
        Guard.NotNull(dataType);
        Guard.NotNull(shape);
        // A descriptor with impossible metadata (negative dims, reversed byte
        // range) must never escape into downstream weight-loading code, where
        // it would surface as overflow or out-of-range reads.
        if (beginByte < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(beginByte), beginByte, "Byte offsets must be non-negative.");
        }

        if (endByte < beginByte)
        {
            throw new ArgumentOutOfRangeException(nameof(endByte), endByte, "endByte must be greater than or equal to beginByte.");
        }

        for (var i = 0; i < shape.Count; i++)
        {
            if (shape[i] < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(shape), shape[i], $"Shape dimension {i} cannot be negative.");
            }
        }

        Name = name;
        DataType = dataType;
        Shape = shape;
        BeginByte = beginByte;
        EndByte = endByte;
    }

    /// <summary>Gets the tensor name.</summary>
    public string Name { get; }

    /// <summary>Gets the safetensors dtype string.</summary>
    public string DataType { get; }

    /// <summary>Gets the tensor shape.</summary>
    public IReadOnlyList<long> Shape { get; }

    /// <summary>Gets the start offset within the data section (inclusive).</summary>
    public long BeginByte { get; }

    /// <summary>Gets the end offset within the data section (exclusive).</summary>
    public long EndByte { get; }

    /// <summary>Gets the number of bytes occupied by this tensor.</summary>
    public long ByteLength => EndByte - BeginByte;

    /// <summary>Gets the total element count (product of <see cref="Shape"/>; 1 for a scalar).</summary>
    /// <exception cref="OverflowException">Thrown when the dimension product exceeds <see cref="long.MaxValue"/>.</exception>
    public long ElementCount
    {
        get
        {
            long count = 1;
            foreach (var dimension in Shape)
            {
                count = checked(count * dimension);
            }

            return count;
        }
    }
}
