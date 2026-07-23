namespace AiDotNet.Data.Collation;

/// <summary>
/// Packs variable-length sequences into a contiguous tensor without padding, along with
/// sequence lengths for reconstruction.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Packed sequences are more memory-efficient than padded batches because they don't store
/// padding tokens. The output contains a flat data tensor of all sequences concatenated,
/// plus a lengths tensor indicating how many elements belong to each sequence.
/// This is equivalent to PyTorch's pack_padded_sequence.
/// </para>
/// <para>
/// Sequences are sorted by length (descending) for efficient RNN processing.
/// </para>
/// </remarks>
public class PackedSequenceCollateFunction<T> : ICollateFunction<Tensor<T>, PackedSequenceBatch<T>>
{
    private readonly bool _sortByLength;

    /// <summary>
    /// Creates a new packed sequence collate function.
    /// </summary>
    /// <param name="sortByLength">Whether to sort sequences by length descending. Default is true.</param>
    public PackedSequenceCollateFunction(bool sortByLength = true)
    {
        _sortByLength = sortByLength;
    }

    /// <inheritdoc/>
    public PackedSequenceBatch<T> Collate(IReadOnlyList<Tensor<T>> samples)
    {
        if (samples.Count == 0)
            throw new ArgumentException("Cannot collate an empty sample list.", nameof(samples));

        // Build index-length pairs
        var indexLengths = new (int Index, int Length)[samples.Count];
        int totalElements = 0;
        for (int i = 0; i < samples.Count; i++)
        {
            int len = samples[i].Data.Length;
            indexLengths[i] = (i, len);
            totalElements += len;
        }

        // Sort by length descending if requested
        if (_sortByLength)
        {
            Array.Sort(indexLengths, (a, b) => b.Length.CompareTo(a.Length));
        }

        // Pack all data into a single flat tensor
        var packedData = new T[totalElements];
        var packedSpan = packedData.AsSpan();
        var lengths = new int[samples.Count];
        var sortedIndices = new int[samples.Count];

        int offset = 0;
        for (int i = 0; i < samples.Count; i++)
        {
            int srcIdx = indexLengths[i].Index;
            int len = indexLengths[i].Length;

            samples[srcIdx].Data.Span.Slice(0, len).CopyTo(packedSpan.Slice(offset, len));
            lengths[i] = len;
            sortedIndices[i] = srcIdx;
            offset += len;
        }

        return new PackedSequenceBatch<T>(
            new Tensor<T>(packedData, new[] { totalElements }),
            lengths,
            sortedIndices);
    }
}

/// <summary>
/// Represents a batch of packed (non-padded) variable-length sequences.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class PackedSequenceBatch<T>
{
    /// <summary>
    /// The flat tensor containing all sequence data concatenated.
    /// </summary>
    public Tensor<T> Data { get; }

    /// <summary>
    /// The length of each sequence in the batch (in sorted order).
    /// </summary>
    public int[] Lengths { get; }

    /// <summary>
    /// Maps sorted position back to original sample index.
    /// </summary>
    public int[] SortedIndices { get; }

    /// <summary>
    /// Number of sequences in the batch.
    /// </summary>
    public int BatchSize => Lengths.Length;

    /// <summary>
    /// Creates a new packed sequence batch.
    /// </summary>
    public PackedSequenceBatch(Tensor<T> data, int[] lengths, int[] sortedIndices)
    {
        Data = data;
        Lengths = lengths;
        SortedIndices = sortedIndices;
    }
}
