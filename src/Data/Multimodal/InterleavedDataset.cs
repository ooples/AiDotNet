namespace AiDotNet.Data.Multimodal;

/// <summary>
/// A dataset where each sample is an interleaved sequence of modality segments,
/// such as alternating image-text-image-text in vision-language models.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Interleaved sequences are common in modern multimodal models like GPT-4V, Gemini, and Flamingo,
/// where images and text are interspersed in a single sequence. Each segment has a modality type
/// and position, enabling the model to process mixed-modality input natively.
/// </para>
/// <para><b>For Beginners:</b> Think of a web page with alternating text paragraphs and images.
/// This dataset represents each "page" as a sequence of typed segments:
/// <code>
/// var sample = new InterleavedSequence&lt;float&gt;();
/// sample.AddSegment(ModalityType.Text, introTokens, "intro");
/// sample.AddSegment(ModalityType.Image, photo1Tensor, "photo1");
/// sample.AddSegment(ModalityType.Text, captionTokens, "caption");
/// sample.AddSegment(ModalityType.Image, photo2Tensor, "photo2");
///
/// var dataset = new InterleavedDataset&lt;float&gt;();
/// dataset.Add(sample);
/// </code>
/// </para>
/// </remarks>
public class InterleavedDataset<T>
{
    private readonly List<InterleavedSequence<T>> _sequences = new List<InterleavedSequence<T>>();

    /// <summary>
    /// Gets the total number of interleaved sequences in the dataset.
    /// </summary>
    public int Count => _sequences.Count;

    /// <summary>
    /// Gets a sequence at the specified index.
    /// </summary>
    public InterleavedSequence<T> this[int index] => _sequences[index];

    /// <summary>
    /// Adds an interleaved sequence to the dataset.
    /// </summary>
    /// <param name="sequence">The sequence to add.</param>
    public void Add(InterleavedSequence<T> sequence)
    {
        if (sequence is null)
            throw new ArgumentNullException(nameof(sequence));

        _sequences.Add(sequence);
    }

    /// <summary>
    /// Adds multiple sequences to the dataset.
    /// </summary>
    public void AddRange(IEnumerable<InterleavedSequence<T>> sequences)
    {
        if (sequences is null)
            throw new ArgumentNullException(nameof(sequences));

        foreach (var seq in sequences)
        {
            Add(seq);
        }
    }

    /// <summary>
    /// Gets the maximum number of segments across all sequences.
    /// </summary>
    public int MaxSegmentCount
    {
        get
        {
            int max = 0;
            foreach (var seq in _sequences)
            {
                if (seq.SegmentCount > max)
                    max = seq.SegmentCount;
            }
            return max;
        }
    }

    /// <summary>
    /// Extracts all segments of a given modality type across all sequences.
    /// </summary>
    /// <param name="modality">The modality to extract.</param>
    /// <returns>All matching segments with their sequence index.</returns>
    public IEnumerable<(int SequenceIndex, int SegmentIndex, InterleavedSegment<T> Segment)> GetSegmentsByModality(
        ModalityType modality)
    {
        for (int seqIdx = 0; seqIdx < _sequences.Count; seqIdx++)
        {
            var seq = _sequences[seqIdx];
            for (int segIdx = 0; segIdx < seq.SegmentCount; segIdx++)
            {
                var segment = seq[segIdx];
                if (segment.Modality == modality)
                {
                    yield return (seqIdx, segIdx, segment);
                }
            }
        }
    }
}

/// <summary>
/// A single interleaved sequence containing ordered segments of different modalities.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class InterleavedSequence<T>
{
    private readonly List<InterleavedSegment<T>> _segments = new List<InterleavedSegment<T>>();

    /// <summary>
    /// Gets the number of segments in this sequence.
    /// </summary>
    public int SegmentCount => _segments.Count;

    /// <summary>
    /// Gets the segments in this sequence.
    /// </summary>
    public IReadOnlyList<InterleavedSegment<T>> Segments => _segments;

    /// <summary>
    /// Gets a segment at the specified position.
    /// </summary>
    public InterleavedSegment<T> this[int index] => _segments[index];

    /// <summary>
    /// Gets or sets an optional label tensor for this sequence.
    /// </summary>
    public Tensor<T>? Label { get; set; }

    /// <summary>
    /// Gets or sets an optional class index for this sequence.
    /// </summary>
    public int ClassIndex { get; set; } = -1;

    /// <summary>
    /// Adds a segment to the end of the sequence.
    /// </summary>
    /// <param name="modality">The modality type of this segment.</param>
    /// <param name="data">The data tensor for this segment.</param>
    /// <param name="key">Optional key for identification.</param>
    public void AddSegment(ModalityType modality, Tensor<T> data, string? key = null)
    {
        _segments.Add(new InterleavedSegment<T>(modality, data, _segments.Count, key));
    }

    /// <summary>
    /// Gets all segment modality types in order.
    /// </summary>
    /// <returns>Ordered array of modality types.</returns>
    public ModalityType[] GetModalitySequence()
    {
        var result = new ModalityType[_segments.Count];
        for (int i = 0; i < _segments.Count; i++)
        {
            result[i] = _segments[i].Modality;
        }
        return result;
    }

    /// <summary>
    /// Gets all segments of a specific modality type.
    /// </summary>
    /// <param name="modality">The modality type to filter by.</param>
    /// <returns>Matching segments in order.</returns>
    public IEnumerable<InterleavedSegment<T>> GetByModality(ModalityType modality)
    {
        return _segments.Where(s => s.Modality == modality);
    }
}

/// <summary>
/// A single segment within an interleaved sequence.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class InterleavedSegment<T>
{
    /// <summary>
    /// Gets the modality type of this segment.
    /// </summary>
    public ModalityType Modality { get; }

    /// <summary>
    /// Gets the data tensor for this segment.
    /// </summary>
    public Tensor<T> Data { get; }

    /// <summary>
    /// Gets the position of this segment within its parent sequence (0-based).
    /// </summary>
    public int Position { get; }

    /// <summary>
    /// Gets an optional key for this segment.
    /// </summary>
    public string Key { get; }

    /// <summary>
    /// Creates a new interleaved segment.
    /// </summary>
    /// <param name="modality">The modality type.</param>
    /// <param name="data">The data tensor.</param>
    /// <param name="position">Position within the sequence.</param>
    /// <param name="key">Optional identifier key.</param>
    public InterleavedSegment(ModalityType modality, Tensor<T> data, int position, string? key = null)
    {
        Modality = modality;
        Guard.NotNull(data);
        Data = data;
        Position = position;
        Key = key ?? $"{modality.ToString().ToLowerInvariant()}_{position}";
    }
}
