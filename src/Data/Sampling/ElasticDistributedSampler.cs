using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Sampling;

/// <summary>
/// Distributed sampler that evenly divides data across multiple workers with elastic scaling support.
/// </summary>
/// <remarks>
/// <para>
/// Ensures each distributed worker sees a non-overlapping subset of the data each epoch.
/// Supports dynamic resizing when workers join or leave (elastic training).
/// Each worker gets dataset_size / num_replicas samples per epoch.
/// </para>
/// </remarks>
public class ElasticDistributedSampler : DataSamplerBase
{
    private readonly ElasticDistributedSamplerOptions _options;
    private int _numReplicas;
    private int _rank;
    private int _samplesPerReplica;

    /// <summary>
    /// Creates a new elastic distributed sampler.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public ElasticDistributedSampler(ElasticDistributedSamplerOptions options)
        : base(options.Seed)
    {
        _options = options;
        _numReplicas = options.NumReplicas;
        _rank = options.Rank;
        _samplesPerReplica = ComputeSamplesPerReplica();
    }

    /// <inheritdoc/>
    public override int Length => _samplesPerReplica;

    /// <summary>
    /// Gets the total dataset size.
    /// </summary>
    public int DatasetSize => _options.DatasetSize;

    /// <summary>
    /// Gets the current number of replicas.
    /// </summary>
    public int NumReplicas => _numReplicas;

    /// <summary>
    /// Gets the current rank.
    /// </summary>
    public int Rank => _rank;

    /// <summary>
    /// Dynamically updates the number of replicas and rank for elastic scaling.
    /// </summary>
    /// <param name="numReplicas">New number of replicas.</param>
    /// <param name="rank">New rank for this worker.</param>
    public void Rescale(int numReplicas, int rank)
    {
        if (numReplicas <= 0)
            throw new ArgumentOutOfRangeException(nameof(numReplicas), "Must be positive.");
        if (rank < 0 || rank >= numReplicas)
            throw new ArgumentOutOfRangeException(nameof(rank), $"Must be in [0, {numReplicas - 1}].");

        _numReplicas = numReplicas;
        _rank = rank;
        _samplesPerReplica = ComputeSamplesPerReplica();
    }

    /// <inheritdoc/>
    protected override IEnumerable<int> GetIndicesCore()
    {
        // Generate all indices
        int[] indices = CreateSequentialIndices(_options.DatasetSize);

        if (_options.Shuffle)
        {
            // Use epoch-dependent seed for reproducible shuffling across replicas.
            // When no seed is set, use a large prime multiplier to differentiate runs.
            int baseSeed = _options.Seed ?? Environment.TickCount;
            var epochRandom = RandomHelper.CreateSeededRandom(CurrentEpoch + baseSeed);
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = epochRandom.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        // Pad if needed so all replicas get same count
        int totalSize = _samplesPerReplica * _numReplicas;
        var padded = new List<int>(indices);
        while (padded.Count < totalSize)
        {
            padded.Add(indices[padded.Count % _options.DatasetSize]);
        }

        // Slice for this rank: interleaved distribution
        for (int i = _rank; i < totalSize; i += _numReplicas)
        {
            yield return padded[i];
        }
    }

    private int ComputeSamplesPerReplica()
    {
        if (_options.DropLast)
            return _options.DatasetSize / _numReplicas;
        return (int)Math.Ceiling((double)_options.DatasetSize / _numReplicas);
    }
}
