using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Preprocessing.ImbalancedLearning;

/// <summary>
/// Base class for undersampling strategies that reduce majority class samples.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Undersampling strategies reduce the number of majority class samples to achieve
/// a more balanced dataset. Unlike oversampling, no synthetic samples are created.
/// </para>
/// <para>
/// <b>For Beginners:</b> If you have 1000 "normal" samples and 50 "fraud" samples,
/// undersampling might reduce the "normal" samples to 50-100 to balance the classes.
///
/// Advantages of undersampling:
/// - Reduces training time (fewer samples)
/// - Simpler than synthetic generation
/// - Works well with large datasets
///
/// Disadvantages:
/// - Loses potentially useful information
/// - Can underfit if too aggressive
/// - Not suitable for small datasets
///
/// Different undersampling strategies choose which samples to remove:
/// - Random: Remove majority samples randomly
/// - Tomek Links: Remove majority samples that form Tomek links with minority
/// - ENN: Remove samples misclassified by nearest neighbors
/// - NearMiss: Keep majority samples closest to minority samples
/// </para>
/// </remarks>
public abstract class UndersamplingBase<T> : IResamplingStrategy<T>
{
    /// <summary>
    /// Numeric operations helper for generic math.
    /// </summary>
    protected readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Random number generator.
    /// </summary>
    protected readonly Random Random;

    /// <summary>
    /// The target ratio of minority to majority class samples.
    /// </summary>
    protected readonly double SamplingStrategy;

    /// <summary>
    /// Statistics about the last resampling operation.
    /// </summary>
    protected ResamplingStatistics<T>? LastStatistics;

    /// <summary>
    /// Gets the name of this undersampling strategy.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Initializes a new instance of the UndersamplingBase class.
    /// </summary>
    /// <param name="samplingStrategy">Target ratio of minority to majority (1.0 for balanced).</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - samplingStrategy = 1.0: Reduce majority to same size as minority
    /// - samplingStrategy = 0.5: Reduce majority to 2x size of minority
    /// - Lower values = more aggressive undersampling
    /// </para>
    /// </remarks>
    protected UndersamplingBase(double samplingStrategy = 1.0, int? seed = null)
    {
        if (samplingStrategy <= 0 || samplingStrategy > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(samplingStrategy),
                "Sampling strategy must be between 0 (exclusive) and 1 (inclusive).");
        }

        SamplingStrategy = samplingStrategy;
        Random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Resamples the dataset by removing majority samples.
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <param name="y">The class labels.</param>
    /// <returns>The resampled features and labels.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method:
    /// 1. Identifies the majority and minority classes
    /// 2. Calculates how many majority samples to keep
    /// 3. Selects which majority samples to keep (strategy-specific)
    /// 4. Returns the reduced dataset
    /// </para>
    /// </remarks>
    public virtual (Matrix<T> resampledX, Vector<T> resampledY) Resample(Matrix<T> x, Vector<T> y)
    {
        // Analyze class distribution
        var classCounts = GetClassCounts(y);
        int minorityCount = classCounts.Values.Min();
        T minorityClass = classCounts.First(kvp => kvp.Value == minorityCount).Key;
        T majorityClass = classCounts.First(kvp => kvp.Value != minorityCount).Key;
        int majorityCount = classCounts[majorityClass];

        // Initialize statistics
        LastStatistics = new ResamplingStatistics<T>
        {
            TotalOriginalSamples = x.Rows
        };

        foreach (var kvp in classCounts)
        {
            LastStatistics.OriginalClassCounts[kvp.Key] = kvp.Value;
        }

        // Calculate target majority count
        int targetMajorityCount = (int)Math.Ceiling(minorityCount / SamplingStrategy);
        int samplesToRemove = Math.Max(0, majorityCount - targetMajorityCount);

        // Get indices
        var minorityIndices = GetClassIndices(y, minorityClass);
        var majorityIndices = GetClassIndices(y, majorityClass);

        // Select majority samples to keep
        List<int> majorityIndicesToKeep;
        if (samplesToRemove > 0 && majorityIndices.Count > targetMajorityCount)
        {
            majorityIndicesToKeep = SelectSamplesToKeep(x, y, majorityIndices, minorityIndices, targetMajorityCount);
        }
        else
        {
            majorityIndicesToKeep = majorityIndices;
        }

        // Combine indices to keep
        var indicesToKeep = new List<int>();
        indicesToKeep.AddRange(minorityIndices);
        indicesToKeep.AddRange(majorityIndicesToKeep);
        indicesToKeep.Sort();

        // Build result matrices
        var resampledX = new Matrix<T>(indicesToKeep.Count, x.Columns);
        var resampledY = new Vector<T>(indicesToKeep.Count);

        for (int i = 0; i < indicesToKeep.Count; i++)
        {
            int originalIdx = indicesToKeep[i];
            for (int j = 0; j < x.Columns; j++)
            {
                resampledX[i, j] = x[originalIdx, j];
            }
            resampledY[i] = y[originalIdx];
        }

        // Update statistics
        LastStatistics.TotalResampledSamples = resampledX.Rows;
        LastStatistics.SamplesAddedPerClass[minorityClass] = 0;
        LastStatistics.SamplesAddedPerClass[majorityClass] = 0;
        LastStatistics.SamplesRemovedPerClass[minorityClass] = 0;
        LastStatistics.SamplesRemovedPerClass[majorityClass] = majorityIndices.Count - majorityIndicesToKeep.Count;
        LastStatistics.ResampledClassCounts[minorityClass] = minorityIndices.Count;
        LastStatistics.ResampledClassCounts[majorityClass] = majorityIndicesToKeep.Count;

        return (resampledX, resampledY);
    }

    /// <summary>
    /// Selects which majority samples to keep.
    /// </summary>
    /// <param name="x">The full feature matrix.</param>
    /// <param name="y">The class labels.</param>
    /// <param name="majorityIndices">Indices of majority class samples.</param>
    /// <param name="minorityIndices">Indices of minority class samples.</param>
    /// <param name="targetCount">Number of majority samples to keep.</param>
    /// <returns>Indices of majority samples to keep.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each undersampling strategy implements this differently:
    /// - Random: Pick randomly
    /// - NearMiss: Pick samples closest to minority
    /// - Tomek/ENN: Remove specific "problematic" samples
    /// </para>
    /// </remarks>
    protected abstract List<int> SelectSamplesToKeep(
        Matrix<T> x,
        Vector<T> y,
        List<int> majorityIndices,
        List<int> minorityIndices,
        int targetCount);

    /// <summary>
    /// Gets the count of samples per class.
    /// </summary>
    protected NumericDictionary<T, int> GetClassCounts(Vector<T> y)
    {
        var counts = new NumericDictionary<T, int>();

        for (int i = 0; i < y.Length; i++)
        {
            if (!counts.TryGetValue(y[i], out int count))
            {
                count = 0;
            }
            counts[y[i]] = count + 1;
        }

        return counts;
    }

    /// <summary>
    /// Gets the indices of samples belonging to a specific class.
    /// </summary>
    protected List<int> GetClassIndices(Vector<T> y, T targetClass)
    {
        var indices = new List<int>();
        for (int i = 0; i < y.Length; i++)
        {
            if (NumOps.Compare(y[i], targetClass) == 0)
            {
                indices.Add(i);
            }
        }
        return indices;
    }

    /// <summary>
    /// Computes the Euclidean distance between two vectors.
    /// </summary>
    protected T EuclideanDistance(Vector<T> a, Vector<T> b)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }
        return NumOps.Sqrt(sum);
    }

    /// <summary>
    /// Gets statistics about the last resampling operation.
    /// </summary>
    public ResamplingStatistics<T> GetStatistics()
    {
        return LastStatistics ?? new ResamplingStatistics<T>();
    }

}
