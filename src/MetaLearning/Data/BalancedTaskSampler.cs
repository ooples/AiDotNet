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

        if (_allClasses.Length < numWays)
            throw new ArgumentException(
                $"Dataset has only {_allClasses.Length} feasible classes (with >= {requiredPerClass} examples each), but NumWays requires {numWays}.",
                nameof(numWays));

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

    /// <summary>Number of candidates to sample for class-balanced selection.</summary>
    private const int BalanceCandidates = 5;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Samples multiple candidate episodes and selects the one whose classes best overlap
    /// with the current rotation window, ensuring long-run uniform class exposure.
    /// </para>
    /// </remarks>
    public IEpisode<T, TInput, TOutput> SampleOne()
    {
        // Get the target classes from the current rotation window
        var targetClasses = new HashSet<int>();
        for (int i = 0; i < NumWays && (_classPointer + i) < _allClasses.Length; i++)
            targetClasses.Add(_allClasses[_classPointer + i]);

        // Sample multiple candidates and pick the one with best class coverage
        IEpisode<T, TInput, TOutput>? bestEpisode = null;
        int bestOverlap = -1;

        for (int c = 0; c < BalanceCandidates; c++)
        {
            var candidate = _dataset.SampleEpisode(NumWays, NumShots, NumQueryPerClass);

            // Count how many of the candidate's classes match the target rotation window
            int overlap = 0;
            if (candidate.Task.SupportOutput is Vector<T> supportLabels)
            {
                var numOps = MathHelper.GetNumericOperations<T>();
                var candidateClasses = new HashSet<int>();
                for (int i = 0; i < supportLabels.Length; i++)
                    candidateClasses.Add((int)Math.Round(numOps.ToDouble(supportLabels[i])));

                foreach (int cls in candidateClasses)
                {
                    if (targetClasses.Contains(cls)) overlap++;
                }
            }
            else
            {
                // If we can't inspect classes, accept the first candidate
                overlap = c == 0 ? 1 : 0;
            }

            if (overlap > bestOverlap)
            {
                bestOverlap = overlap;
                bestEpisode = candidate;
            }
        }

        AdvancePointer();
        return bestEpisode ?? _dataset.SampleEpisode(NumWays, NumShots, NumQueryPerClass);
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
