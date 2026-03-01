using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// Supports the Meta-Dataset benchmark format (Triantafillou et al., 2020): a multi-domain
/// evaluation protocol with variable-way variable-shot task sampling. Wraps multiple
/// <see cref="IMetaDataset{T, TInput, TOutput}"/> instances and samples episodes from a
/// randomly chosen domain, or from a specific domain on request.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The Meta-Dataset benchmark tests how well a model generalizes
/// across multiple very different domains (e.g., birds, textures, aircraft, fungi). This class
/// lets you combine several datasets and sample tasks from any of them.</para>
/// <para><b>Reference:</b> Meta-Dataset: A Dataset of Datasets for Learning to Learn
/// from Few Examples (Triantafillou et al., ICLR 2020).</para>
/// </remarks>
public class MetaDatasetFormat<T, TInput, TOutput> : IMetaDataset<T, TInput, TOutput>
{
    private readonly List<IMetaDataset<T, TInput, TOutput>> _domains;
    private readonly List<string> _domainNames;
    private Random _rng;

    /// <inheritdoc/>
    public string Name => "MetaDataset";

    /// <inheritdoc/>
    public int TotalClasses
    {
        get
        {
            int total = 0;
            foreach (var d in _domains) total += d.TotalClasses;
            return total;
        }
    }

    /// <inheritdoc/>
    public int TotalExamples
    {
        get
        {
            int total = 0;
            foreach (var d in _domains) total += d.TotalExamples;
            return total;
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<int, int> ClassExampleCounts
    {
        get
        {
            var combined = new Dictionary<int, int>();
            int offset = 0;
            foreach (var d in _domains)
            {
                foreach (var kvp in d.ClassExampleCounts)
                    combined[offset + kvp.Key] = kvp.Value;
                offset += d.TotalClasses;
            }
            return combined;
        }
    }

    /// <summary>
    /// Gets the number of domains in this multi-domain dataset.
    /// </summary>
    public int DomainCount => _domains.Count;

    /// <summary>
    /// Gets the domain names.
    /// </summary>
    public IReadOnlyList<string> DomainNames => _domainNames;

    /// <summary>
    /// Creates a multi-domain meta-dataset from multiple individual datasets.
    /// </summary>
    /// <param name="domains">Individual datasets, one per domain.</param>
    /// <param name="domainNames">Optional names for each domain. If null, uses "Domain_0", "Domain_1", etc.</param>
    /// <param name="seed">Optional random seed.</param>
    public MetaDatasetFormat(
        IReadOnlyList<IMetaDataset<T, TInput, TOutput>> domains,
        IReadOnlyList<string>? domainNames = null,
        int? seed = null)
    {
        if (domains.Count == 0)
            throw new ArgumentException("At least one domain is required.", nameof(domains));

        _domains = new List<IMetaDataset<T, TInput, TOutput>>(domains);
        _domainNames = new List<string>();
        for (int i = 0; i < domains.Count; i++)
        {
            _domainNames.Add(domainNames != null && i < domainNames.Count ? domainNames[i] : $"Domain_{i}");
        }
        _rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc/>
    public IEpisode<T, TInput, TOutput> SampleEpisode(int numWays, int numShots, int numQueryPerClass)
    {
        int domainIdx = SelectDomain(numWays, numShots, numQueryPerClass);
        var episode = _domains[domainIdx].SampleEpisode(numWays, numShots, numQueryPerClass);
        episode.Difficulty ??= 0.5;
        // Wrap with domain metadata
        return new Episode<T, TInput, TOutput>(
            episode.Task,
            domain: _domainNames[domainIdx],
            difficulty: episode.Difficulty,
            metadata: new Dictionary<string, object> { ["source_domain_index"] = domainIdx });
    }

    /// <inheritdoc/>
    public IReadOnlyList<IEpisode<T, TInput, TOutput>> SampleEpisodes(
        int count, int numWays, int numShots, int numQueryPerClass)
    {
        var episodes = new List<IEpisode<T, TInput, TOutput>>(count);
        for (int i = 0; i < count; i++)
            episodes.Add(SampleEpisode(numWays, numShots, numQueryPerClass));
        return episodes;
    }

    /// <summary>
    /// Samples an episode from a specific domain by index.
    /// </summary>
    /// <param name="domainIndex">The zero-based domain index.</param>
    /// <param name="numWays">Number of classes per episode.</param>
    /// <param name="numShots">Number of support examples per class.</param>
    /// <param name="numQueryPerClass">Number of query examples per class.</param>
    /// <returns>An episode from the specified domain.</returns>
    public IEpisode<T, TInput, TOutput> SampleFromDomain(int domainIndex, int numWays, int numShots, int numQueryPerClass)
    {
        if (domainIndex < 0 || domainIndex >= _domains.Count)
            throw new ArgumentOutOfRangeException(nameof(domainIndex));

        var episode = _domains[domainIndex].SampleEpisode(numWays, numShots, numQueryPerClass);
        return new Episode<T, TInput, TOutput>(
            episode.Task,
            domain: _domainNames[domainIndex],
            difficulty: episode.Difficulty,
            metadata: new Dictionary<string, object> { ["source_domain_index"] = domainIndex });
    }

    /// <inheritdoc/>
    public void SetSeed(int seed)
    {
        _rng = RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < _domains.Count; i++)
            _domains[i].SetSeed(seed + i);
    }

    /// <inheritdoc/>
    public bool SupportsConfiguration(int numWays, int numShots, int numQueryPerClass)
    {
        foreach (var d in _domains)
            if (d.SupportsConfiguration(numWays, numShots, numQueryPerClass))
                return true;
        return false;
    }

    private int SelectDomain(int numWays, int numShots, int numQueryPerClass)
    {
        // Build list of feasible domains, select uniformly at random
        var feasible = new List<int>();
        for (int i = 0; i < _domains.Count; i++)
        {
            if (_domains[i].SupportsConfiguration(numWays, numShots, numQueryPerClass))
                feasible.Add(i);
        }
        if (feasible.Count == 0)
            throw new InvalidOperationException(
                $"No domain supports the requested configuration ({numWays}-way {numShots}-shot {numQueryPerClass}-query).");
        return feasible[_rng.Next(feasible.Count)];
    }
}
