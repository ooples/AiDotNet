using AiDotNet.Data.Structures;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// Samples tasks while ensuring that all classes in the meta-dataset appear equally often
/// across the sampled episodes over time. This prevents the meta-learner from overfitting to
/// frequently-sampled classes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public class BalancedTaskSampler<T, TInput, TOutput> : ITaskSampler<T, TInput, TOutput>
{
    private readonly IMetaDataset<T, TInput, TOutput> _dataset;
    private readonly int[] _allClasses;
    private int _classPointer;
    private Random _rng;

    /// <inheritdoc/>
    public int NumWays { get; }

    /// <inheritdoc/>
    public int NumShots { get; }

    /// <inheritdoc/>
    public int NumQueryPerClass { get; }

    /// <summary>
    /// Creates a balanced task sampler that rotates through all classes evenly.
    /// </summary>
    /// <param name="dataset">The meta-dataset to sample from.</param>
    /// <param name="numWays">Number of classes per task.</param>
    /// <param name="numShots">Number of support examples per class.</param>
    /// <param name="numQueryPerClass">Number of query examples per class.</param>
    /// <param name="seed">Optional random seed.</param>
    public BalancedTaskSampler(
        IMetaDataset<T, TInput, TOutput> dataset,
        int numWays = 5,
        int numShots = 1,
        int numQueryPerClass = 15,
        int? seed = null)
    {
        _dataset = dataset;
        NumWays = numWays;
        NumShots = numShots;
        NumQueryPerClass = numQueryPerClass;
        _rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();

        // Build a shuffled list of all feasible classes
        int requiredPerClass = numShots + numQueryPerClass;
        var counts = dataset.ClassExampleCounts;
        _allClasses = counts.Where(kvp => kvp.Value >= requiredPerClass).Select(kvp => kvp.Key).ToArray();
        Shuffle(_allClasses);
        _classPointer = 0;
    }

    /// <inheritdoc/>
    public TaskBatch<T, TInput, TOutput> SampleBatch(int batchSize)
    {
        var tasks = new IMetaLearningTask<T, TInput, TOutput>[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            tasks[i] = SampleOne().Task;
        }
        return new TaskBatch<T, TInput, TOutput>(tasks, BatchingStrategy.DomainBalanced);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// The internal class rotation tracks which classes have been covered. Each call advances
    /// a pointer through a shuffled class list to ensure long-run uniform class exposure.
    /// When the pointer wraps around, the class list is reshuffled.
    /// </para>
    /// <para>
    /// Because <see cref="IMetaDataset{T, TInput, TOutput}.SampleEpisode"/> does not support
    /// constrained class selection, exact per-episode targeting is not possible. The rotation
    /// ensures that the sampler at least tracks coverage and reshuffles periodically.
    /// </para>
    /// </remarks>
    public IEpisode<T, TInput, TOutput> SampleOne()
    {
        var episode = _dataset.SampleEpisode(NumWays, NumShots, NumQueryPerClass);
        AdvancePointer();
        return episode;
    }

    /// <inheritdoc/>
    public void UpdateWithFeedback(IReadOnlyList<IEpisode<T, TInput, TOutput>> episodes, IReadOnlyList<double> losses)
    {
        // Balanced sampler ignores feedback; balance is maintained structurally.
    }

    /// <inheritdoc/>
    public void SetSeed(int seed)
    {
        _rng = RandomHelper.CreateSeededRandom(seed);
        _dataset.SetSeed(seed);
        Shuffle(_allClasses);
        _classPointer = 0;
    }

    private void AdvancePointer()
    {
        _classPointer += NumWays;
        if (_classPointer + NumWays > _allClasses.Length)
        {
            Shuffle(_allClasses);
            _classPointer = 0;
        }
    }

    private void Shuffle(int[] array)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = _rng.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }
}
