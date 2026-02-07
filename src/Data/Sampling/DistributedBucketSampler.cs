using AiDotNet.Interfaces;

namespace AiDotNet.Data.Sampling;

/// <summary>
/// Combines distributed partitioning with bucket batching for efficient distributed NLP training.
/// </summary>
/// <remarks>
/// <para>
/// Each rank receives a disjoint partition of the dataset (like DistributedSampler), then
/// within that partition, samples are grouped by length into buckets for efficient batching
/// (like BucketBatchSampler). This minimizes padding waste in distributed sequence model training.
/// </para>
/// </remarks>
public class DistributedBucketSampler : DataSamplerBase, IBatchSampler
{
    private readonly int[] _sampleLengths;
    private readonly int _numReplicas;
    private readonly int _rank;
    private readonly int _numBuckets;
    private readonly bool _shuffle;
    private readonly int? _baseSeed;
    private int _samplesPerRank;

    /// <inheritdoc/>
    public int BatchSize { get; set; }

    /// <inheritdoc/>
    public bool DropLast { get; set; }

    /// <inheritdoc/>
    public override int Length => _samplesPerRank;

    /// <summary>
    /// Creates a new distributed bucket sampler.
    /// </summary>
    /// <param name="sampleLengths">The length of each sample.</param>
    /// <param name="numReplicas">Total number of distributed processes.</param>
    /// <param name="rank">The rank of the current process (0-based).</param>
    /// <param name="batchSize">Number of samples per batch.</param>
    /// <param name="numBuckets">Number of length buckets. Default is 10.</param>
    /// <param name="shuffle">Whether to shuffle. Default is true.</param>
    /// <param name="dropLast">Whether to drop incomplete batches.</param>
    /// <param name="seed">Optional base seed for reproducibility.</param>
    public DistributedBucketSampler(
        int[] sampleLengths,
        int numReplicas,
        int rank,
        int batchSize,
        int numBuckets = 10,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null)
        : base(seed)
    {
        if (sampleLengths is null || sampleLengths.Length == 0)
            throw new ArgumentException("Sample lengths cannot be null or empty.", nameof(sampleLengths));
        if (numReplicas < 1)
            throw new ArgumentOutOfRangeException(nameof(numReplicas), "Number of replicas must be at least 1.");
        if (rank < 0 || rank >= numReplicas)
            throw new ArgumentOutOfRangeException(nameof(rank), $"Rank must be in [0, {numReplicas - 1}].");
        if (batchSize < 1)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be at least 1.");

        _sampleLengths = sampleLengths;
        _numReplicas = numReplicas;
        _rank = rank;
        BatchSize = batchSize;
        _numBuckets = numBuckets;
        _shuffle = shuffle;
        DropLast = dropLast;
        _baseSeed = seed;

        _samplesPerRank = (int)Math.Ceiling((double)sampleLengths.Length / numReplicas);
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
        // Get this rank's portion of indices
        int[] rankIndices = GetRankIndices();

        // Create a bucket sampler for this rank's data
        int[] rankLengths = new int[rankIndices.Length];
        for (int i = 0; i < rankIndices.Length; i++)
        {
            rankLengths[i] = _sampleLengths[rankIndices[i]];
        }

        // Sort by length within rank
        int[] sortedPositions = new int[rankIndices.Length];
        for (int i = 0; i < sortedPositions.Length; i++) sortedPositions[i] = i;
        Array.Sort(sortedPositions, (a, b) => rankLengths[a].CompareTo(rankLengths[b]));

        // Divide into buckets
        int samplesPerBucket = Math.Max(1, sortedPositions.Length / _numBuckets);
        var buckets = new List<List<int>>();

        for (int start = 0; start < sortedPositions.Length; start += samplesPerBucket)
        {
            int end = Math.Min(start + samplesPerBucket, sortedPositions.Length);
            var bucket = new List<int>(end - start);
            for (int i = start; i < end; i++)
            {
                bucket.Add(rankIndices[sortedPositions[i]]);
            }
            buckets.Add(bucket);
        }

        // Shuffle within buckets
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

        // Yield batches
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

    private int[] GetRankIndices()
    {
        int totalSize = _sampleLengths.Length;
        int[] allIndices = CreateSequentialIndices(totalSize);

        // Deterministic shuffle based on epoch
        if (_shuffle)
        {
            int epochSeed = _baseSeed.HasValue
                ? _baseSeed.Value + CurrentEpoch
                : CurrentEpoch;
            var epochRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(epochSeed);
            for (int i = allIndices.Length - 1; i > 0; i--)
            {
                int j = epochRandom.Next(i + 1);
                (allIndices[i], allIndices[j]) = (allIndices[j], allIndices[i]);
            }
        }

        // Pad to make evenly divisible
        int paddedTotal = (int)Math.Ceiling((double)totalSize / _numReplicas) * _numReplicas;
        if (paddedTotal > totalSize)
        {
            int[] padded = new int[paddedTotal];
            Array.Copy(allIndices, padded, totalSize);
            for (int i = totalSize; i < paddedTotal; i++)
            {
                padded[i] = allIndices[i % totalSize];
            }
            allIndices = padded;
        }

        // Extract this rank's slice
        int perRank = paddedTotal / _numReplicas;
        int startIdx = _rank * perRank;
        int[] rankIndices = new int[perRank];
        Array.Copy(allIndices, startIdx, rankIndices, 0, perRank);

        _samplesPerRank = perRank;
        return rankIndices;
    }
}
