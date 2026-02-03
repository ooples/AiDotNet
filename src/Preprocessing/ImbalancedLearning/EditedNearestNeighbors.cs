using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Preprocessing.ImbalancedLearning;

/// <summary>
/// Implements Edited Nearest Neighbors (ENN) undersampling for handling imbalanced datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// ENN removes samples whose class label differs from the majority of their k nearest
/// neighbors. This removes noisy and borderline samples from the majority class.
/// </para>
/// <para>
/// <b>For Beginners:</b> ENN uses a simple rule to decide which samples to remove:
///
/// "If a sample's neighbors mostly disagree with its label, remove it."
///
/// For example, if a majority class sample has 3 nearest neighbors:
/// - 2 are minority class
/// - 1 is majority class
/// Then the sample is "misclassified" by its neighbors and is removed.
///
/// This is like asking: "Would a K-NN classifier with K=3 correctly classify this sample?"
/// If no, the sample is probably noisy or on the wrong side of the boundary.
///
/// Visual example:
/// ```
/// Before:  M M M m M m m m m
///                ^
///          This M surrounded by m's would be removed
///
/// After:   M M M   m m m m m
///          Cleaner boundary
/// ```
///
/// When to use:
/// - Data cleaning before training
/// - In combination with SMOTE (SMOTE + ENN = SMOTEENN)
/// - When you want to remove ambiguous/noisy samples
///
/// References:
/// - Wilson (1972). "Asymptotic Properties of Nearest Neighbor Rules Using Edited Data"
/// </para>
/// </remarks>
public class EditedNearestNeighbors<T> : IResamplingStrategy<T>
{
    /// <summary>
    /// Numeric operations helper for generic math.
    /// </summary>
    private readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Number of nearest neighbors to consider.
    /// </summary>
    private readonly int _kNeighbors;

    /// <summary>
    /// The editing kind to use.
    /// </summary>
    private readonly ENNKind _kind;

    /// <summary>
    /// Statistics about the last resampling operation.
    /// </summary>
    private ResamplingStatistics<T>? _lastStatistics;

    /// <summary>
    /// Gets the name of this undersampling strategy.
    /// </summary>
    public string Name => "EditedNearestNeighbors";

    /// <summary>
    /// Initializes a new instance of the EditedNearestNeighbors class.
    /// </summary>
    /// <param name="kNeighbors">Number of nearest neighbors to use. Default is 3.</param>
    /// <param name="kind">The editing kind. Default is All (remove if any neighbor disagrees).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The parameters control how aggressive the cleaning is:
    ///
    /// - kNeighbors: More neighbors = more context for decision
    ///   - k=3: Remove if 2+ neighbors disagree (sensitive)
    ///   - k=5: Remove if 3+ neighbors disagree (moderate)
    ///
    /// - kind:
    ///   - All: Remove if majority of neighbors disagree (standard)
    ///   - Mode: Remove if most common neighbor class differs from sample class
    /// </para>
    /// </remarks>
    public EditedNearestNeighbors(int kNeighbors = 3, ENNKind kind = ENNKind.All)
    {
        if (kNeighbors < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(kNeighbors),
                "Number of neighbors must be at least 1.");
        }

        _kNeighbors = kNeighbors;
        _kind = kind;
    }

    /// <summary>
    /// Resamples the dataset by removing samples misclassified by their neighbors.
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <param name="y">The class labels.</param>
    /// <returns>The resampled features and labels.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method:
    /// 1. For each majority sample, finds its k nearest neighbors
    /// 2. Checks if the majority of neighbors have a different class
    /// 3. If so, removes the sample
    /// 4. Keeps all minority samples (we don't want to lose any)
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

        int effectiveK = Math.Min(_kNeighbors, x.Rows - 1);

        // Find samples to remove
        var samplesToRemove = new HashSet<int>();
        var majorityIndices = GetClassIndices(y, majorityClass);
        var allIndices = Enumerable.Range(0, x.Rows).ToList();

        foreach (int idx in majorityIndices)
        {
            // Find k nearest neighbors
            var neighbors = FindKNearestNeighbors(x, idx, allIndices, effectiveK);

            // Count neighbor classes
            int minorityNeighborCount = 0;
            foreach (int neighborIdx in neighbors)
            {
                if (NumOps.Compare(y[neighborIdx], minorityClass) == 0)
                {
                    minorityNeighborCount++;
                }
            }

            // Decide whether to remove based on kind
            bool shouldRemove;
            if (_kind == ENNKind.All)
            {
                // Remove if majority of neighbors are minority class
                shouldRemove = minorityNeighborCount > neighbors.Count / 2;
            }
            else
            {
                // Mode: Remove if most common neighbor class is different
                shouldRemove = minorityNeighborCount >= (neighbors.Count + 1) / 2;
            }

            if (shouldRemove)
            {
                samplesToRemove.Add(idx);
            }
        }

        // Build list of indices to keep
        var indicesToKeep = new List<int>();
        for (int i = 0; i < x.Rows; i++)
        {
            if (!samplesToRemove.Contains(i))
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
        var minorityIndices = GetClassIndices(y, minorityClass);
        _lastStatistics.TotalResampledSamples = resampledX.Rows;
        _lastStatistics.SamplesAddedPerClass[minorityClass] = 0;
        _lastStatistics.SamplesAddedPerClass[majorityClass] = 0;
        _lastStatistics.SamplesRemovedPerClass[minorityClass] = 0;
        _lastStatistics.SamplesRemovedPerClass[majorityClass] = samplesToRemove.Count;
        _lastStatistics.ResampledClassCounts[minorityClass] = minorityIndices.Count;
        _lastStatistics.ResampledClassCounts[majorityClass] = majorityIndices.Count - samplesToRemove.Count;

        return (resampledX, resampledY);
    }

    /// <summary>
    /// Finds the k nearest neighbors of a sample.
    /// </summary>
    private List<int> FindKNearestNeighbors(Matrix<T> x, int sampleIndex, List<int> candidateIndices, int k)
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

        return distances
            .OrderBy(d => NumOps.ToDouble(d.distance))
            .Take(Math.Min(k, distances.Count))
            .Select(d => d.index)
            .ToList();
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

/// <summary>
/// Specifies the editing kind for ENN.
/// </summary>
public enum ENNKind
{
    /// <summary>
    /// Remove if majority of neighbors are of different class.
    /// </summary>
    All,

    /// <summary>
    /// Remove if the mode (most common) neighbor class is different.
    /// </summary>
    Mode
}
