using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Preprocessing.ImbalancedLearning;

/// <summary>
/// Implements Tomek Links undersampling for handling imbalanced datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Tomek Links removes majority class samples that form "Tomek links" with minority
/// samples. A Tomek link is a pair of samples from different classes that are each
/// other's nearest neighbor.
/// </para>
/// <para>
/// <b>For Beginners:</b> A Tomek link is a special relationship between two samples:
///
/// 1. Sample A (minority class) and Sample B (majority class)
/// 2. A's nearest neighbor is B
/// 3. B's nearest neighbor is A
/// 4. They are each other's closest sample across classes!
///
/// Why Tomek links are important:
/// - They represent borderline or noisy samples
/// - Removing the majority sample from a Tomek link cleans the decision boundary
/// - It helps the classifier focus on clearer cases
///
/// Visual example:
/// ```
/// Before:  M . . . . m M . . . m . . . . M
///          ^         ^
///          These two might be a Tomek link
///
/// After:   . . . . . m . . . . m . . . . M
///          Majority sample removed, cleaner boundary
/// ```
///
/// M = majority, m = minority
///
/// When to use:
/// - Data cleaning before training
/// - In combination with oversampling (SMOTE + Tomek)
/// - When you want minimal but targeted undersampling
///
/// References:
/// - Tomek (1976). "Two Modifications of CNN"
/// </para>
/// </remarks>
public class TomekLinks<T> : IResamplingStrategy<T>
{
    /// <summary>
    /// Numeric operations helper for generic math.
    /// </summary>
    private readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Statistics about the last resampling operation.
    /// </summary>
    private ResamplingStatistics<T>? _lastStatistics;

    /// <summary>
    /// Gets the name of this undersampling strategy.
    /// </summary>
    public string Name => "TomekLinks";

    /// <summary>
    /// Initializes a new instance of the TomekLinks class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Tomek Links doesn't have a sampling strategy parameter
    /// because it only removes specific samples (those forming Tomek links), not a
    /// target ratio. The amount of undersampling depends on the data.
    /// </para>
    /// </remarks>
    public TomekLinks()
    {
    }

    /// <summary>
    /// Resamples the dataset by removing Tomek links.
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <param name="y">The class labels.</param>
    /// <returns>The resampled features and labels.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method:
    /// 1. Finds all Tomek links in the dataset
    /// 2. Removes the majority class sample from each Tomek link
    /// 3. Keeps all minority samples and non-Tomek majority samples
    /// </para>
    /// </remarks>
    public (Matrix<T> resampledX, Vector<T> resampledY) Resample(Matrix<T> x, Vector<T> y)
    {
        var classCounts = GetClassCounts(y);
        int minorityCount = classCounts.Values.Min();
        T minorityClass = classCounts.First(kvp => kvp.Value == minorityCount).Key;
        T majorityClass = classCounts.First(kvp => kvp.Value != minorityCount).Key;

        // Initialize statistics
        _lastStatistics = new ResamplingStatistics<T>
        {
            TotalOriginalSamples = x.Rows
        };

        foreach (var kvp in classCounts)
        {
            _lastStatistics.OriginalClassCounts[kvp.Key] = kvp.Value;
        }

        // Find all Tomek links
        var tomekLinksToRemove = new HashSet<int>();
        var minorityIndices = GetClassIndices(y, minorityClass);
        var majorityIndices = GetClassIndices(y, majorityClass);

        foreach (int minIdx in minorityIndices)
        {
            // Find nearest neighbor of this minority sample
            int nearestNeighbor = FindNearestNeighbor(x, minIdx);

            // Check if it's a majority sample
            if (majorityIndices.Contains(nearestNeighbor))
            {
                // Check if the minority sample is also the majority sample's nearest neighbor
                int majNearestNeighbor = FindNearestNeighbor(x, nearestNeighbor);

                if (majNearestNeighbor == minIdx)
                {
                    // This is a Tomek link - mark majority sample for removal
                    tomekLinksToRemove.Add(nearestNeighbor);
                }
            }
        }

        // Build list of indices to keep
        var indicesToKeep = new List<int>();
        for (int i = 0; i < x.Rows; i++)
        {
            if (!tomekLinksToRemove.Contains(i))
            {
                indicesToKeep.Add(i);
            }
        }

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
        _lastStatistics.TotalResampledSamples = resampledX.Rows;
        _lastStatistics.SamplesAddedPerClass[minorityClass] = 0;
        _lastStatistics.SamplesAddedPerClass[majorityClass] = 0;
        _lastStatistics.SamplesRemovedPerClass[minorityClass] = 0;
        _lastStatistics.SamplesRemovedPerClass[majorityClass] = tomekLinksToRemove.Count;
        _lastStatistics.ResampledClassCounts[minorityClass] = minorityIndices.Count;
        _lastStatistics.ResampledClassCounts[majorityClass] = majorityIndices.Count - tomekLinksToRemove.Count;

        return (resampledX, resampledY);
    }

    /// <summary>
    /// Finds the nearest neighbor of a sample.
    /// </summary>
    private int FindNearestNeighbor(Matrix<T> x, int sampleIndex)
    {
        var sample = x.GetRow(sampleIndex);
        int nearestIdx = -1;
        T minDistance = NumOps.MaxValue;

        for (int i = 0; i < x.Rows; i++)
        {
            if (i == sampleIndex) continue;

            T distance = EuclideanDistance(sample, x.GetRow(i));
            if (NumOps.Compare(distance, minDistance) < 0)
            {
                minDistance = distance;
                nearestIdx = i;
            }
        }

        return nearestIdx;
    }

    /// <summary>
    /// Computes the Euclidean distance between two vectors.
    /// </summary>
    private T EuclideanDistance(Vector<T> a, Vector<T> b)
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
    /// Gets the count of samples per class.
    /// </summary>
    private NumericDictionary<T, int> GetClassCounts(Vector<T> y)
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
    private List<int> GetClassIndices(Vector<T> y, T targetClass)
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
    /// Gets statistics about the last resampling operation.
    /// </summary>
    public ResamplingStatistics<T> GetStatistics()
    {
        return _lastStatistics ?? new ResamplingStatistics<T>();
    }

}
