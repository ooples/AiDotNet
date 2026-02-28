using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// Abstract base class for meta-datasets that generate episodes on-the-fly.
/// Subclasses provide the data; this base class handles episode construction and validation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public abstract class MetaDatasetBase<T, TInput, TOutput> : IMetaDataset<T, TInput, TOutput>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Random number generator for sampling.
    /// </summary>
    protected Random Rng;

    /// <inheritdoc/>
    public abstract string Name { get; }

    /// <inheritdoc/>
    public abstract int TotalClasses { get; }

    /// <inheritdoc/>
    public abstract int TotalExamples { get; }

    /// <inheritdoc/>
    public abstract IReadOnlyDictionary<int, int> ClassExampleCounts { get; }

    /// <summary>
    /// Initializes the base meta-dataset with an optional seed.
    /// </summary>
    /// <param name="seed">Optional seed for reproducibility.</param>
    protected MetaDatasetBase(int? seed = null)
    {
        Rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc/>
    public virtual IEpisode<T, TInput, TOutput> SampleEpisode(int numWays, int numShots, int numQueryPerClass)
    {
        ValidateConfiguration(numWays, numShots, numQueryPerClass);
        var task = SampleTaskCore(numWays, numShots, numQueryPerClass);
        return new Episode<T, TInput, TOutput>(task);
    }

    /// <inheritdoc/>
    public virtual IReadOnlyList<IEpisode<T, TInput, TOutput>> SampleEpisodes(
        int count, int numWays, int numShots, int numQueryPerClass)
    {
        ValidateConfiguration(numWays, numShots, numQueryPerClass);
        var episodes = new List<IEpisode<T, TInput, TOutput>>(count);
        for (int i = 0; i < count; i++)
        {
            var task = SampleTaskCore(numWays, numShots, numQueryPerClass);
            episodes.Add(new Episode<T, TInput, TOutput>(task));
        }
        return episodes;
    }

    /// <inheritdoc/>
    public void SetSeed(int seed)
    {
        Rng = RandomHelper.CreateSeededRandom(seed);
    }

    /// <inheritdoc/>
    public virtual bool SupportsConfiguration(int numWays, int numShots, int numQueryPerClass)
    {
        if (numWays < 1 || numShots < 1 || numQueryPerClass < 1) return false;
        if (numWays > TotalClasses) return false;

        int requiredPerClass = numShots + numQueryPerClass;
        var counts = ClassExampleCounts;
        int feasibleClasses = 0;
        foreach (var kvp in counts)
        {
            if (kvp.Value >= requiredPerClass) feasibleClasses++;
        }
        return feasibleClasses >= numWays;
    }

    /// <summary>
    /// Core method that samples a single meta-learning task. Subclasses must implement this.
    /// </summary>
    /// <param name="numWays">Number of classes.</param>
    /// <param name="numShots">Number of support examples per class.</param>
    /// <param name="numQueryPerClass">Number of query examples per class.</param>
    /// <returns>A meta-learning task.</returns>
    protected abstract IMetaLearningTask<T, TInput, TOutput> SampleTaskCore(
        int numWays, int numShots, int numQueryPerClass);

    /// <summary>
    /// Validates that the requested configuration is feasible for this dataset.
    /// </summary>
    protected void ValidateConfiguration(int numWays, int numShots, int numQueryPerClass)
    {
        if (!SupportsConfiguration(numWays, numShots, numQueryPerClass))
        {
            throw new ArgumentException(
                $"Configuration ({numWays}-way {numShots}-shot with {numQueryPerClass} queries) is not feasible for dataset '{Name}' " +
                $"with {TotalClasses} classes.");
        }
    }

    /// <summary>
    /// Selects <paramref name="count"/> random distinct integers from [0, <paramref name="max"/>).
    /// </summary>
    protected int[] SampleWithoutReplacement(int max, int count)
    {
        if (count > max)
            throw new ArgumentException($"Cannot sample {count} items from range [0, {max}).");

        var selected = new HashSet<int>();
        while (selected.Count < count)
        {
            selected.Add(Rng.Next(max));
        }
        return selected.ToArray();
    }
}
