using AiDotNet.Helpers;

namespace AiDotNet.Data.Pipeline;

/// <summary>
/// Mixes multiple data sources with configurable weights for multi-domain training.
/// </summary>
/// <remarks>
/// <para>
/// Combines multiple datasets into a single stream with weighted sampling.
/// Each batch is drawn from sources according to their mixing weights.
/// Commonly used for training on mixtures of domains (e.g., code + text + math).
/// </para>
/// </remarks>
/// <typeparam name="TItem">The type of items produced by each source.</typeparam>
public class MultiSourceMixer<TItem>
{
    private readonly MultiSourceMixerOptions _options;
    private readonly Random _random;
    private readonly double[] _normalizedWeights;
    private readonly double[] _cumulativeWeights;

    /// <summary>
    /// Creates a new multi-source mixer.
    /// </summary>
    /// <param name="numSources">Number of data sources.</param>
    /// <param name="options">Configuration options.</param>
    public MultiSourceMixer(int numSources, MultiSourceMixerOptions? options = null)
    {
        if (numSources <= 0)
            throw new ArgumentOutOfRangeException(nameof(numSources), "Number of sources must be positive.");

        _options = options ?? new MultiSourceMixerOptions();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize weights
        double[] weights;
        if (_options.Weights != null && _options.Weights.Length == numSources)
        {
            weights = (double[])_options.Weights.Clone();
        }
        else
        {
            weights = new double[numSources];
            for (int i = 0; i < numSources; i++)
                weights[i] = 1.0;
        }

        // Normalize
        double sum = 0;
        foreach (double w in weights)
            sum += w;

        _normalizedWeights = new double[numSources];
        for (int i = 0; i < numSources; i++)
            _normalizedWeights[i] = weights[i] / sum;

        // Cumulative
        _cumulativeWeights = new double[numSources];
        double cumulative = 0;
        for (int i = 0; i < numSources; i++)
        {
            cumulative += _normalizedWeights[i];
            _cumulativeWeights[i] = cumulative;
        }
        _cumulativeWeights[numSources - 1] = 1.0;
    }

    /// <summary>
    /// Selects which source to draw the next item from.
    /// </summary>
    /// <returns>Index of the selected source.</returns>
    public int SelectSource()
    {
        double r = _random.NextDouble();
        for (int i = 0; i < _cumulativeWeights.Length; i++)
        {
            if (r <= _cumulativeWeights[i])
                return i;
        }
        return _cumulativeWeights.Length - 1;
    }

    /// <summary>
    /// Mixes items from multiple sources using weighted sampling.
    /// </summary>
    /// <param name="sources">The data sources as enumerables.</param>
    /// <param name="totalItems">Total number of items to produce.</param>
    /// <returns>Interleaved items from the sources.</returns>
    public IEnumerable<TItem> Mix(IReadOnlyList<IEnumerator<TItem>> sources, int totalItems)
    {
        if (sources == null || sources.Count == 0)
            throw new ArgumentException("Sources must not be null or empty.", nameof(sources));
        if (totalItems < 0)
            throw new ArgumentOutOfRangeException(nameof(totalItems), "Total items must be non-negative.");

        var active = new bool[sources.Count];
        for (int i = 0; i < sources.Count; i++)
            active[i] = true;

        for (int produced = 0; produced < totalItems; produced++)
        {
            int sourceIdx = SelectSource();

            // Try to advance the selected source
            if (!active[sourceIdx] || !sources[sourceIdx].MoveNext())
            {
                if (_options.StopOnShortestSource)
                    yield break;

                active[sourceIdx] = false;

                // Find any active source
                bool foundActive = false;
                for (int attempt = 0; attempt < sources.Count; attempt++)
                {
                    int altIdx = (sourceIdx + attempt + 1) % sources.Count;
                    if (active[altIdx] && sources[altIdx].MoveNext())
                    {
                        yield return sources[altIdx].Current;
                        foundActive = true;
                        break;
                    }
                }

                if (!foundActive) yield break;
            }
            else
            {
                yield return sources[sourceIdx].Current;
            }
        }
    }

    /// <summary>
    /// Gets the normalized weight for a source.
    /// </summary>
    /// <param name="sourceIndex">Index of the source.</param>
    /// <returns>Normalized weight in [0, 1].</returns>
    public double GetWeight(int sourceIndex)
    {
        if (sourceIndex < 0 || sourceIndex >= _normalizedWeights.Length)
            throw new ArgumentOutOfRangeException(nameof(sourceIndex));
        return _normalizedWeights[sourceIndex];
    }

    /// <summary>
    /// Gets the number of sources.
    /// </summary>
    public int NumSources => _normalizedWeights.Length;
}
