using AiDotNet.Interfaces;

namespace AiDotNet.Data.Sampling;

/// <summary>
/// Groups sequences by length into buckets, then batches within each bucket to minimize padding.
/// </summary>
/// <remarks>
/// <para>
/// Bucket batching is essential for efficient NLP training. By grouping sequences of similar
/// lengths together, padding waste is minimized and GPU utilization is maximized.
/// </para>
/// <para>
/// The sampler sorts samples by their length, divides them into buckets of roughly equal size,
/// shuffles within each bucket, and yields batches from each bucket.
/// </para>
/// </remarks>
public class BucketBatchSampler : DataSamplerBase, IBatchSampler
{
    private readonly int[] _sampleLengths;
    private readonly int _numBuckets;

    /// <inheritdoc/>
    public int BatchSize { get; set; }

    /// <inheritdoc/>
    public bool DropLast { get; set; }

    /// <inheritdoc/>
    public override int Length => _sampleLengths.Length;

    /// <summary>
    /// Creates a new bucket batch sampler.
    /// </summary>
    /// <param name="sampleLengths">The length of each sample (e.g., number of tokens).</param>
    /// <param name="batchSize">Number of samples per batch.</param>
    /// <param name="numBuckets">Number of length buckets. Default is 10.</param>
    /// <param name="dropLast">Whether to drop the last incomplete batch in each bucket.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public BucketBatchSampler(
        int[] sampleLengths,
        int batchSize,
        int numBuckets = 10,
        bool dropLast = false,
        int? seed = null)
        : base(seed)
    {
        if (sampleLengths is null || sampleLengths.Length == 0)
            throw new ArgumentException("Sample lengths cannot be null or empty.", nameof(sampleLengths));
        if (batchSize < 1)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be at least 1.");
        if (numBuckets < 1)
            throw new ArgumentOutOfRangeException(nameof(numBuckets), "Number of buckets must be at least 1.");

        _sampleLengths = sampleLengths;
        BatchSize = batchSize;
        _numBuckets = numBuckets;
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
        // Sort indices by sample length
        int[] sortedIndices = CreateSequentialIndices(_sampleLengths.Length);
        Array.Sort(sortedIndices, (a, b) => _sampleLengths[a].CompareTo(_sampleLengths[b]));

        // Divide into buckets
        int samplesPerBucket = Math.Max(1, sortedIndices.Length / _numBuckets);
        var buckets = new List<List<int>>();

        for (int start = 0; start < sortedIndices.Length; start += samplesPerBucket)
        {
            int end = Math.Min(start + samplesPerBucket, sortedIndices.Length);
            var bucket = new List<int>(end - start);
            for (int i = start; i < end; i++)
            {
                bucket.Add(sortedIndices[i]);
            }
            buckets.Add(bucket);
        }

        // Shuffle within each bucket
        foreach (var bucket in buckets)
        {
            for (int i = bucket.Count - 1; i > 0; i--)
            {
                int j = Random.Next(i + 1);
                int temp = bucket[i];
                bucket[i] = bucket[j];
                bucket[j] = temp;
            }
        }

        // Shuffle bucket order
        for (int i = buckets.Count - 1; i > 0; i--)
        {
            int j = Random.Next(i + 1);
            var temp = buckets[i];
            buckets[i] = buckets[j];
            buckets[j] = temp;
        }

        // Yield batches from each bucket
        foreach (var bucket in buckets)
        {
            for (int start = 0; start + BatchSize <= bucket.Count; start += BatchSize)
            {
                int[] batch = new int[BatchSize];
                for (int i = 0; i < BatchSize; i++)
                {
                    batch[i] = bucket[start + i];
                }
                yield return batch;
            }

            // Handle remainder
            int remainder = bucket.Count % BatchSize;
            if (remainder > 0 && !DropLast)
            {
                int start = bucket.Count - remainder;
                int[] batch = new int[remainder];
                for (int i = 0; i < remainder; i++)
                {
                    batch[i] = bucket[start + i];
                }
                yield return batch;
            }
        }
    }
}
