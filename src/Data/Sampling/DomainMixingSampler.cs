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
    private readonly double[] _cumulativeWeights;
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
        if (domainSizes == null || domainSizes.Length == 0)
            throw new ArgumentException("domainSizes must not be null or empty.", nameof(domainSizes));
        if (domainWeights == null || domainWeights.Length == 0)
            throw new ArgumentException("domainWeights must not be null or empty.", nameof(domainWeights));
        if (domainSizes.Length != domainWeights.Length)
            throw new ArgumentException("domainSizes and domainWeights must have the same length.");
        for (int i = 0; i < domainSizes.Length; i++)
        {
            if (domainSizes[i] <= 0)
                throw new ArgumentOutOfRangeException(nameof(domainSizes), $"Domain size at index {i} must be positive.");
            if (domainWeights[i] < 0 || double.IsNaN(domainWeights[i]) || double.IsInfinity(domainWeights[i]))
                throw new ArgumentOutOfRangeException(nameof(domainWeights), $"Domain weight at index {i} must be a non-negative finite number.");
        }

        // Defensive copies to prevent external mutation
        _domainSizes = (int[])domainSizes.Clone();

        // Normalize weights
        double weightSum = domainWeights.Sum();
        if (weightSum <= 0)
            throw new ArgumentException("domainWeights must sum to a positive value.", nameof(domainWeights));
        _domainWeights = domainWeights.Select(w => w / weightSum).ToArray();

        // Pre-compute cumulative distribution for domain selection
        _cumulativeWeights = new double[_domainWeights.Length];
        _cumulativeWeights[0] = _domainWeights[0];
        for (int i = 1; i < _domainWeights.Length; i++)
            _cumulativeWeights[i] = _cumulativeWeights[i - 1] + _domainWeights[i];
        _cumulativeWeights[_domainWeights.Length - 1] = 1.0; // Ensure no floating-point gap

        // Compute cumulative offsets with overflow check
        _domainOffsets = new int[_domainSizes.Length];
        long offset = 0;
        for (int i = 0; i < _domainSizes.Length; i++)
        {
            if (offset > int.MaxValue)
                throw new OverflowException($"Total domain sizes exceed int.MaxValue at domain index {i}.");
            _domainOffsets[i] = (int)offset;
            offset += _domainSizes[i];
        }
        if (offset > int.MaxValue)
            throw new OverflowException($"Total domain sizes ({offset}) exceed int.MaxValue.");
        _totalSize = (int)offset;
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

        for (int i = 0; i < _totalSize; i++)
        {
            // Select domain based on pre-computed cumulative weights
            double r = Random.NextDouble();
            int domain = _domainWeights.Length - 1; // default to last domain (handles floating-point edge)
            for (int d = 0; d < _cumulativeWeights.Length; d++)
            {
                if (r <= _cumulativeWeights[d])
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
