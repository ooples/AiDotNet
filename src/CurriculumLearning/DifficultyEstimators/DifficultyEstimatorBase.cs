using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.CurriculumLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.CurriculumLearning.DifficultyEstimators;

/// <summary>
/// Base class for difficulty estimators.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This base class provides common functionality for all
/// difficulty estimators. It handles the mechanics of computing difficulty scores
/// and sorting samples by difficulty.</para>
/// </remarks>
public abstract class DifficultyEstimatorBase<T, TInput, TOutput> : IDifficultyEstimator<T, TInput, TOutput>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the name of the difficulty estimator.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Gets whether this estimator requires the model to estimate difficulty.
    /// </summary>
    public abstract bool RequiresModel { get; }

    /// <summary>
    /// Gets or sets whether to cache difficulty scores.
    /// </summary>
    public bool CacheScores { get; set; } = true;

    /// <summary>
    /// Gets the cached difficulty scores (if caching is enabled).
    /// </summary>
    protected Vector<T>? CachedScores { get; set; }

    /// <summary>
    /// Gets whether scores have been computed and cached.
    /// </summary>
    protected bool HasCachedScores => CachedScores != null;

    /// <summary>
    /// Estimates the difficulty of a single sample.
    /// </summary>
    public abstract T EstimateDifficulty(
        TInput input,
        TOutput expectedOutput,
        IFullModel<T, TInput, TOutput>? model = null);

    /// <summary>
    /// Estimates difficulty scores for all samples in a dataset.
    /// </summary>
    public virtual Vector<T> EstimateDifficulties(
        IDataset<T, TInput, TOutput> dataset,
        IFullModel<T, TInput, TOutput>? model = null)
    {
        if (dataset is null) throw new ArgumentNullException(nameof(dataset));

        if (RequiresModel && model == null)
        {
            throw new ArgumentNullException(nameof(model),
                $"The {Name} estimator requires a model to estimate difficulties.");
        }

        // Return cached scores if available
        if (CacheScores && HasCachedScores && CachedScores!.Length == dataset.Count)
        {
            return CachedScores;
        }

        var difficulties = new T[dataset.Count];

        // Default implementation: estimate each sample individually
        // Subclasses can override for batch optimization
        for (int i = 0; i < dataset.Count; i++)
        {
            var sample = dataset.GetSample(i);
            difficulties[i] = EstimateDifficulty(sample.Input, sample.Output, model);
        }

        var result = new Vector<T>(difficulties);

        if (CacheScores)
        {
            CachedScores = result;
        }

        return result;
    }

    /// <summary>
    /// Updates the difficulty estimator based on training progress.
    /// </summary>
    public virtual void Update(int epoch, IFullModel<T, TInput, TOutput> model)
    {
        // Default: clear cache so difficulties will be recalculated
        CachedScores = null;
    }

    /// <summary>
    /// Resets the estimator to its initial state.
    /// </summary>
    public virtual void Reset()
    {
        CachedScores = null;
    }

    /// <summary>
    /// Gets the indices of samples sorted by difficulty (easy to hard).
    /// </summary>
    public virtual int[] GetSortedIndices(Vector<T> difficulties)
    {
        if (difficulties is null) throw new ArgumentNullException(nameof(difficulties));

        // Create index array
        var indices = new int[difficulties.Length];
        for (int i = 0; i < indices.Length; i++)
        {
            indices[i] = i;
        }

        // Sort indices by difficulty values (ascending = easy to hard)
        Array.Sort(indices, (a, b) => NumOps.Compare(difficulties[a], difficulties[b]));

        return indices;
    }

    /// <summary>
    /// Normalizes difficulty scores to [0, 1] range.
    /// </summary>
    /// <param name="difficulties">The difficulty scores to normalize.</param>
    /// <returns>Normalized difficulty scores.</returns>
    protected virtual Vector<T> NormalizeDifficulties(Vector<T> difficulties)
    {
        if (difficulties.Length == 0)
        {
            return difficulties;
        }

        var min = difficulties.Minimum();
        var max = difficulties.Max();

        // If all values are the same, return zeros
        if (NumOps.Equals(min, max))
        {
            return new Vector<T>(Enumerable.Repeat(NumOps.Zero, difficulties.Length));
        }

        var range = NumOps.Subtract(max, min);
        var normalized = new Vector<T>(difficulties.Length);

        for (int i = 0; i < difficulties.Length; i++)
        {
            normalized[i] = NumOps.Divide(NumOps.Subtract(difficulties[i], min), range);
        }

        return normalized;
    }

    /// <summary>
    /// Computes the mean of a vector.
    /// </summary>
    protected T ComputeMean(Vector<T> values)
    {
        if (values.Length == 0)
        {
            return NumOps.Zero;
        }

        var sum = NumOps.Zero;
        foreach (var value in values)
        {
            sum = NumOps.Add(sum, value);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(values.Length));
    }

    /// <summary>
    /// Computes the standard deviation of a vector.
    /// </summary>
    protected T ComputeStandardDeviation(Vector<T> values, T mean)
    {
        if (values.Length <= 1)
        {
            return NumOps.Zero;
        }

        var sumSquaredDiffs = NumOps.Zero;
        foreach (var value in values)
        {
            var diff = NumOps.Subtract(value, mean);
            sumSquaredDiffs = NumOps.Add(sumSquaredDiffs, NumOps.Multiply(diff, diff));
        }

        var variance = NumOps.Divide(sumSquaredDiffs, NumOps.FromDouble(values.Length - 1));
        return NumOps.Sqrt(variance);
    }
}
