using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Sampling;

/// <summary>
/// Partitions dataset indices across N ranks for distributed (multi-GPU/multi-node) training.
/// </summary>
/// <remarks>
/// <para>
/// Each rank sees a disjoint 1/N subset of the data. Optionally pads the dataset so all ranks
/// receive exactly the same number of samples (required when using AllReduce-style gradient
/// synchronization). Equivalent to PyTorch's DistributedSampler.
/// </para>
/// <para>
/// At each epoch, the sampler shuffles the full dataset deterministically using the epoch number
/// as the seed modifier, ensuring all ranks agree on the same shuffle order.
/// </para>
/// </remarks>
public class DistributedSampler : DataSamplerBase
{
    private readonly int _datasetSize;
    private readonly int _numReplicas;
    private readonly int _rank;
    private readonly bool _shuffle;
    private readonly bool _dropLast;
    private readonly int? _baseSeed;

    /// <summary>
    /// Number of samples assigned to this rank.
    /// </summary>
    public int NumSamplesPerRank { get; private set; }

    /// <inheritdoc/>
    public override int Length => NumSamplesPerRank;

    /// <summary>
    /// Creates a new distributed sampler.
    /// </summary>
    /// <param name="datasetSize">Total number of samples in the dataset.</param>
    /// <param name="numReplicas">Total number of distributed processes (ranks).</param>
    /// <param name="rank">The rank of the current process (0-based).</param>
    /// <param name="shuffle">Whether to shuffle indices each epoch. Default is true.</param>
    /// <param name="dropLast">Whether to drop extra samples to make dataset evenly divisible. Default is false.</param>
    /// <param name="seed">Optional base seed for reproducibility.</param>
    public DistributedSampler(
        int datasetSize,
        int numReplicas,
        int rank,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null)
        : base(seed)
    {
        if (datasetSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(datasetSize), "Dataset size must be positive.");
        if (numReplicas < 1)
            throw new ArgumentOutOfRangeException(nameof(numReplicas), "Number of replicas must be at least 1.");
        if (rank < 0 || rank >= numReplicas)
            throw new ArgumentOutOfRangeException(nameof(rank), $"Rank must be in [0, {numReplicas - 1}].");

        _datasetSize = datasetSize;
        _numReplicas = numReplicas;
        _rank = rank;
        _shuffle = shuffle;
        _dropLast = dropLast;
        _baseSeed = seed;

        ComputeSamplesPerRank();
    }

    /// <inheritdoc/>
    protected override IEnumerable<int> GetIndicesCore()
    {
        int[] indices = CreateSequentialIndices(_datasetSize);

        // Shuffle deterministically based on epoch + seed so all ranks agree
        if (_shuffle)
        {
            int epochSeed = _baseSeed.HasValue
                ? _baseSeed.Value + CurrentEpoch
                : CurrentEpoch;
            var epochRandom = RandomHelper.CreateSeededRandom(epochSeed);
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = epochRandom.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        // Pad if needed (repeat from the beginning)
        int totalSize;
        if (_dropLast)
        {
            totalSize = (_datasetSize / _numReplicas) * _numReplicas;
        }
        else
        {
            totalSize = (int)Math.Ceiling((double)_datasetSize / _numReplicas) * _numReplicas;
        }

        // If we need more indices than dataset size, wrap around
        int[] paddedIndices;
        if (totalSize > _datasetSize)
        {
            paddedIndices = new int[totalSize];
            for (int i = 0; i < totalSize; i++)
            {
                paddedIndices[i] = indices[i % _datasetSize];
            }
        }
        else if (totalSize < _datasetSize)
        {
            paddedIndices = new int[totalSize];
            Array.Copy(indices, paddedIndices, totalSize);
        }
        else
        {
            paddedIndices = indices;
        }

        // Extract this rank's portion
        int samplesPerRank = totalSize / _numReplicas;
        int startIdx = _rank * samplesPerRank;

        for (int i = 0; i < samplesPerRank; i++)
        {
            yield return paddedIndices[startIdx + i];
        }
    }

    private void ComputeSamplesPerRank()
    {
        if (_dropLast)
        {
            NumSamplesPerRank = _datasetSize / _numReplicas;
        }
        else
        {
            NumSamplesPerRank = (int)Math.Ceiling((double)_datasetSize / _numReplicas);
        }
    }
}
