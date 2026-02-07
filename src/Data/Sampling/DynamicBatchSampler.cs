using AiDotNet.Interfaces;

namespace AiDotNet.Data.Sampling;

/// <summary>
/// Creates batches that fit a maximum number of tokens/elements rather than a fixed number of samples.
/// </summary>
/// <remarks>
/// <para>
/// Unlike fixed-size batching (N samples per batch), dynamic batching fits as many samples
/// as possible up to a token/element budget. This maximizes GPU utilization for variable-length
/// inputs by ensuring each batch uses roughly the same amount of memory regardless of sequence lengths.
/// </para>
/// <para>
/// For example, with a max tokens budget of 512: a batch might contain 8 sequences of length 64,
/// or 2 sequences of length 256, or 1 sequence of length 512.
/// </para>
/// </remarks>
public class DynamicBatchSampler : DataSamplerBase, IBatchSampler
{
    private readonly int[] _sampleLengths;
    private readonly int _maxTokensPerBatch;
    private readonly bool _shuffle;

    /// <summary>
    /// Gets or sets the maximum number of samples per batch (upper bound even if token budget allows more).
    /// </summary>
    public int BatchSize { get; set; }

    /// <inheritdoc/>
    public bool DropLast { get; set; }

    /// <inheritdoc/>
    public override int Length => _sampleLengths.Length;

    /// <summary>
    /// Creates a new dynamic batch sampler.
    /// </summary>
    /// <param name="sampleLengths">The length of each sample (e.g., number of tokens).</param>
    /// <param name="maxTokensPerBatch">Maximum total tokens allowed in a single batch.</param>
    /// <param name="maxSamplesPerBatch">Maximum samples per batch (upper bound). Default is int.MaxValue.</param>
    /// <param name="shuffle">Whether to shuffle samples before batching. Default is true.</param>
    /// <param name="dropLast">Whether to drop the last incomplete batch.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public DynamicBatchSampler(
        int[] sampleLengths,
        int maxTokensPerBatch,
        int maxSamplesPerBatch = int.MaxValue,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null)
        : base(seed)
    {
        if (sampleLengths is null || sampleLengths.Length == 0)
            throw new ArgumentException("Sample lengths cannot be null or empty.", nameof(sampleLengths));
        if (maxTokensPerBatch < 1)
            throw new ArgumentOutOfRangeException(nameof(maxTokensPerBatch), "Max tokens per batch must be at least 1.");

        _sampleLengths = sampleLengths;
        _maxTokensPerBatch = maxTokensPerBatch;
        BatchSize = maxSamplesPerBatch;
        _shuffle = shuffle;
        DropLast = dropLast;
    }

    /// <inheritdoc/>
    protected override IEnumerable<int> GetIndicesCore()
    {
        foreach (int[] batch in GetBatchIndices())
        {
            foreach (int idx in batch)
            {
                yield return idx;
            }
        }
    }

    /// <inheritdoc/>
    public IEnumerable<int[]> GetBatchIndices()
    {
        int[] indices = CreateSequentialIndices(_sampleLengths.Length);

        if (_shuffle)
        {
            ShuffleIndices(indices);
        }

        var currentBatch = new List<int>();
        int currentTokens = 0;

        foreach (int idx in indices)
        {
            int sampleLen = _sampleLengths[idx];

            // If adding this sample would exceed the token budget or sample limit, yield current batch
            if (currentBatch.Count > 0 &&
                (currentTokens + sampleLen > _maxTokensPerBatch || currentBatch.Count >= BatchSize))
            {
                yield return currentBatch.ToArray();
                currentBatch.Clear();
                currentTokens = 0;
            }

            // If a single sample exceeds the budget, it gets its own batch
            currentBatch.Add(idx);
            currentTokens += sampleLen;
        }

        // Handle remaining samples
        if (currentBatch.Count > 0 && !DropLast)
        {
            yield return currentBatch.ToArray();
        }
    }
}
