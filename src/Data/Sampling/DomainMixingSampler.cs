namespace AiDotNet.Data.Sampling;

/// <summary>
/// Samples from multiple data domains with configurable mixing ratios for multi-domain LLM training.
/// </summary>
/// <remarks>
/// <para>
/// Domain mixing controls what proportion of each batch comes from different data sources
/// (e.g., 40% web, 30% code, 20% books, 10% academic). This is the technique used
/// by Llama, GPT-4, and other large language models to balance data quality and diversity.
/// </para>
/// </remarks>
public class DomainMixingSampler : DataSamplerBase
{
    private readonly double[] _domainWeights;
    private readonly int[] _domainSizes;
    private readonly int[] _domainOffsets;
    private readonly int _totalSize;

    /// <summary>
    /// Initializes a new domain mixing sampler.
    /// </summary>
    /// <param name="domainSizes">Number of samples in each domain.</param>
    /// <param name="domainWeights">Sampling probability for each domain. Must sum to ~1.0.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public DomainMixingSampler(int[] domainSizes, double[] domainWeights, int? seed = null)
        : base(seed)
    {
        if (domainSizes.Length != domainWeights.Length)
            throw new ArgumentException("domainSizes and domainWeights must have the same length.");

        _domainSizes = domainSizes;

        // Normalize weights
        double weightSum = domainWeights.Sum();
        _domainWeights = domainWeights.Select(w => w / weightSum).ToArray();

        // Compute cumulative offsets
        _domainOffsets = new int[domainSizes.Length];
        int offset = 0;
        for (int i = 0; i < domainSizes.Length; i++)
        {
            _domainOffsets[i] = offset;
            offset += domainSizes[i];
        }
        _totalSize = offset;
    }

    /// <inheritdoc/>
    public override int Length => _totalSize;

    /// <summary>
    /// Returns indices sampled according to domain mixing weights.
    /// </summary>
    /// <returns>Indices respecting domain mixing ratios.</returns>
    protected override IEnumerable<int> GetIndicesCore()
    {
        var indices = new int[_totalSize];

        // Build cumulative distribution for domain selection
        var cumulativeWeights = new double[_domainWeights.Length];
        cumulativeWeights[0] = _domainWeights[0];
        for (int i = 1; i < _domainWeights.Length; i++)
            cumulativeWeights[i] = cumulativeWeights[i - 1] + _domainWeights[i];

        for (int i = 0; i < _totalSize; i++)
        {
            // Select domain based on weights
            double r = Random.NextDouble();
            int domain = 0;
            for (int d = 0; d < cumulativeWeights.Length; d++)
            {
                if (r <= cumulativeWeights[d])
                {
                    domain = d;
                    break;
                }
            }

            // Sample random index within selected domain
            int localIdx = Random.Next(_domainSizes[domain]);
            indices[i] = _domainOffsets[domain] + localIdx;
        }

        return indices;
    }
}
