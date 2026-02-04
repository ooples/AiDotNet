using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Preprocessing.ImbalancedLearning;

/// <summary>
/// Base class for oversampling strategies that create synthetic samples for minority classes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Oversampling strategies increase the number of minority class samples by creating synthetic
/// examples. This helps machine learning models learn to recognize minority classes.
/// </para>
/// <para>
/// <b>For Beginners:</b> When you have imbalanced data (e.g., 1000 "normal" samples but only
/// 50 "fraud" samples), the model often ignores the minority class. Oversampling creates
/// synthetic "fraud" samples so the model sees enough examples to learn the pattern.
///
/// Different oversampling strategies create synthetic samples in different ways:
/// - SMOTE: Creates samples between existing minority samples
/// - ADASYN: Creates more samples in regions where classification is harder
/// - BorderlineSMOTE: Focuses on samples near the decision boundary
///
/// Important: Only oversample the training data! Never oversample test data.
/// </para>
/// </remarks>
public abstract class OversamplingBase<T> : IResamplingStrategy<T>
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
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls how balanced the final dataset will be.
    /// - 1.0 means equal numbers of each class (perfect balance)
    /// - 0.5 means minority will be half the size of majority
    /// - "auto" (default) means balance all classes equally
    /// </para>
    /// </remarks>
    protected readonly double SamplingStrategy;

    /// <summary>
    /// Number of nearest neighbors to use.
    /// </summary>
    protected readonly int KNeighbors;

    /// <summary>
    /// Statistics about the last resampling operation.
    /// </summary>
    protected ResamplingStatistics<T>? LastStatistics;

    /// <summary>
    /// Gets the name of this oversampling strategy.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Initializes a new instance of the OversamplingBase class.
    /// </summary>
    /// <param name="samplingStrategy">Target ratio of minority to majority (1.0 for balanced).</param>
    /// <param name="kNeighbors">Number of nearest neighbors for synthesis.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The main parameters are:
    /// - samplingStrategy: How balanced should the result be? 1.0 means perfect balance.
    /// - kNeighbors: How many similar samples to consider when creating synthetic ones.
    ///   5 is a common default.
    /// - seed: Set this for reproducible results (same seed = same synthetic samples).
    /// </para>
    /// </remarks>
    protected OversamplingBase(double samplingStrategy = 1.0, int kNeighbors = 5, int? seed = null)
    {
        if (samplingStrategy <= 0 || samplingStrategy > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(samplingStrategy),
                "Sampling strategy must be between 0 (exclusive) and 1 (inclusive).");
        }

        if (kNeighbors < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(kNeighbors),
                "Number of neighbors must be at least 1.");
        }

        SamplingStrategy = samplingStrategy;
        KNeighbors = kNeighbors;
        Random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Resamples the dataset by creating synthetic minority samples.
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <param name="y">The class labels.</param>
    /// <returns>The resampled features and labels.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method:
    /// 1. Analyzes your data to find majority and minority classes
    /// 2. Calculates how many synthetic samples are needed
    /// 3. Creates synthetic samples for each minority class
    /// 4. Returns the combined original + synthetic data
    /// </para>
    /// </remarks>
    public virtual (Matrix<T> resampledX, Vector<T> resampledY) Resample(Matrix<T> x, Vector<T> y)
    {
        // Analyze class distribution
        var classCounts = GetClassCounts(y);
        int majorityCount = classCounts.Values.Max();

        // Initialize statistics
        LastStatistics = new ResamplingStatistics<T>
        {
            TotalOriginalSamples = x.Rows
        };

        foreach (var kvp in classCounts)
        {
            LastStatistics.OriginalClassCounts[kvp.Key] = kvp.Value;
        }

        // Collect all samples (original + synthetic)
        var allFeatures = new List<Vector<T>>();
        var allLabels = new List<T>();

        // Add original samples
        for (int i = 0; i < x.Rows; i++)
        {
            allFeatures.Add(x.GetRow(i));
            allLabels.Add(y[i]);
        }

        // Generate synthetic samples for each minority class
        foreach (var kvp in classCounts)
        {
            T classLabel = kvp.Key;
            int classCount = kvp.Value;

            // Calculate target count based on sampling strategy
            int targetCount = (int)(majorityCount * SamplingStrategy);
            int samplesToGenerate = Math.Max(0, targetCount - classCount);

            if (samplesToGenerate > 0)
            {
                // Get indices of this class
                var classIndices = GetClassIndices(y, classLabel);

                // Generate synthetic samples
                var syntheticSamples = GenerateSyntheticSamples(x, classIndices, samplesToGenerate);

                foreach (var sample in syntheticSamples)
                {
                    allFeatures.Add(sample);
                    allLabels.Add(classLabel);
                }

                LastStatistics.SamplesAddedPerClass[classLabel] = syntheticSamples.Count;
            }
            else
            {
                LastStatistics.SamplesAddedPerClass[classLabel] = 0;
            }

            LastStatistics.SamplesRemovedPerClass[classLabel] = 0;
        }

        // Build result matrices
        var resampledX = new Matrix<T>(allFeatures.Count, x.Columns);
        var resampledY = new Vector<T>(allLabels.Count);

        for (int i = 0; i < allFeatures.Count; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                resampledX[i, j] = allFeatures[i][j];
            }
            resampledY[i] = allLabels[i];
        }

        // Update statistics
        LastStatistics.TotalResampledSamples = resampledX.Rows;
        foreach (var kvp in classCounts)
        {
            LastStatistics.ResampledClassCounts[kvp.Key] =
                LastStatistics.OriginalClassCounts[kvp.Key] +
                LastStatistics.SamplesAddedPerClass[kvp.Key];
        }

        return (resampledX, resampledY);
    }

    /// <summary>
    /// Generates synthetic samples for a minority class.
    /// </summary>
    /// <param name="x">The full feature matrix.</param>
    /// <param name="classIndices">Indices of samples belonging to the target class.</param>
    /// <param name="numSamples">Number of synthetic samples to generate.</param>
    /// <returns>List of synthetic sample vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is where the magic happens - each oversampling strategy
    /// implements this differently to create synthetic samples in its own way.
    /// </para>
    /// </remarks>
    protected abstract List<Vector<T>> GenerateSyntheticSamples(Matrix<T> x, List<int> classIndices, int numSamples);

    /// <summary>
    /// Gets the count of samples per class.
    /// </summary>
    /// <param name="y">The class labels.</param>
    /// <returns>Dictionary mapping class labels to counts.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This counts how many samples belong to each class,
    /// which is needed to determine the majority class and calculate how many
    /// synthetic samples to create.
    /// </para>
    /// </remarks>
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
    /// <param name="y">The class labels.</param>
    /// <param name="targetClass">The class to find.</param>
    /// <returns>List of indices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This finds all the positions in the data that belong to
    /// a specific class, so we can work with just those samples when creating
    /// synthetic examples.
    /// </para>
    /// </remarks>
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
    /// <param name="a">First vector.</param>
    /// <param name="b">Second vector.</param>
    /// <returns>The Euclidean distance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This measures how "far apart" two samples are in feature space.
    /// Samples that are close together are similar, samples far apart are different.
    /// This is used to find nearest neighbors.
    /// </para>
    /// </remarks>
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
    /// Finds the k nearest neighbors of a sample within a set of candidates.
    /// </summary>
    /// <param name="x">The full feature matrix.</param>
    /// <param name="sampleIndex">Index of the sample to find neighbors for.</param>
    /// <param name="candidateIndices">Indices of candidate neighbors.</param>
    /// <param name="k">Number of neighbors to find.</param>
    /// <returns>Indices of the k nearest neighbors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This finds the k samples that are most similar to a given sample.
    /// "Similar" means close in feature space (small Euclidean distance).
    /// </para>
    /// </remarks>
    protected List<int> FindKNearestNeighbors(Matrix<T> x, int sampleIndex, List<int> candidateIndices, int k)
    {
        var sample = x.GetRow(sampleIndex);
        var distances = new List<(int index, T distance)>();

        foreach (int candidateIndex in candidateIndices)
        {
            if (candidateIndex != sampleIndex)
            {
                var candidate = x.GetRow(candidateIndex);
                T distance = EuclideanDistance(sample, candidate);
                distances.Add((candidateIndex, distance));
            }
        }

        // Sort by distance and take top k
        var sorted = distances
            .OrderBy(d => NumOps.ToDouble(d.distance))
            .Take(Math.Min(k, distances.Count))
            .Select(d => d.index)
            .ToList();

        return sorted;
    }

    /// <summary>
    /// Gets statistics about the last resampling operation.
    /// </summary>
    /// <returns>Resampling statistics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Call this after Resample() to see what happened -
    /// how many samples were in each class before and after, how many were created.
    /// </para>
    /// </remarks>
    public ResamplingStatistics<T> GetStatistics()
    {
        return LastStatistics ?? new ResamplingStatistics<T>();
    }

}
