using AiDotNet.LinearAlgebra;

namespace AiDotNet.Preprocessing.ImbalancedLearning;

/// <summary>
/// Implements NearMiss undersampling for handling imbalanced datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// NearMiss selects majority class samples based on their distance to minority class samples.
/// It comes in three versions with different selection strategies.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of randomly removing majority samples, NearMiss intelligently
/// selects which ones to keep based on how close they are to minority samples.
///
/// Three versions:
///
/// NearMiss-1: Keep majority samples closest to nearest minority samples
/// - For each majority sample, find distance to closest minority sample
/// - Keep majority samples with smallest such distances
/// - Focuses on boundary samples
///
/// NearMiss-2: Keep majority samples closest to farthest minority samples
/// - For each majority sample, find distance to farthest minority sample
/// - Keep majority samples with smallest such distances
/// - Keeps samples that are "in between" the minority class
///
/// NearMiss-3: Keep majority samples closest to each minority sample
/// - For each minority sample, find k nearest majority neighbors
/// - Keep those majority samples
/// - Ensures every minority sample has nearby majority examples
///
/// Visual intuition:
/// ```
/// m m m . . . M M M M M M M M M
///         ^           ^
///     Boundary      Far from boundary
///
/// NearMiss-1: Keeps M's near the m's (boundary)
/// NearMiss-2: Keeps M's that are somewhat near m's but not too far
/// NearMiss-3: Keeps specific M's nearest to each m
/// ```
///
/// References:
/// - Mani & Zhang (2003). "kNN Approach to Unbalanced Data Distributions"
/// </para>
/// </remarks>
public class NearMiss<T> : UndersamplingBase<T>
{
    private readonly NearMissVersion _version;
    private readonly int _kNeighbors;

    /// <summary>
    /// Gets the name of this undersampling strategy.
    /// </summary>
    public override string Name => $"NearMiss-{(int)_version}";

    /// <summary>
    /// Initializes a new instance of the NearMiss class.
    /// </summary>
    /// <param name="samplingStrategy">Target ratio of minority to majority (1.0 for balanced). Default is 1.0.</param>
    /// <param name="version">NearMiss version to use. Default is Version1.</param>
    /// <param name="kNeighbors">Number of neighbors for version 3. Default is 3.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Example usage:
    ///
    /// <code>
    /// // Default: NearMiss-1 with balanced classes
    /// var nearmiss = new NearMiss&lt;double&gt;();
    ///
    /// // NearMiss-3 which keeps k nearest majority neighbors per minority sample
    /// var nearmiss = new NearMiss&lt;double&gt;(version: NearMissVersion.Version3, kNeighbors: 5);
    ///
    /// // Apply to data
    /// var (newX, newY) = nearmiss.Resample(trainX, trainY);
    /// </code>
    /// </para>
    /// </remarks>
    public NearMiss(
        double samplingStrategy = 1.0,
        NearMissVersion version = NearMissVersion.Version1,
        int kNeighbors = 3,
        int? seed = null)
        : base(samplingStrategy, seed)
    {
        _version = version;
        _kNeighbors = kNeighbors;
    }

    /// <summary>
    /// Selects which majority samples to keep using NearMiss selection.
    /// </summary>
    /// <param name="x">The full feature matrix.</param>
    /// <param name="y">The class labels.</param>
    /// <param name="majorityIndices">Indices of majority class samples.</param>
    /// <param name="minorityIndices">Indices of minority class samples.</param>
    /// <param name="targetCount">Number of majority samples to keep.</param>
    /// <returns>Indices of majority samples to keep.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method selects the "most useful" majority samples
    /// based on the NearMiss version. The idea is that samples near the boundary
    /// are more informative than samples far from minority class.
    /// </para>
    /// </remarks>
    protected override List<int> SelectSamplesToKeep(
        Matrix<T> x,
        Vector<T> y,
        List<int> majorityIndices,
        List<int> minorityIndices,
        int targetCount)
    {
        return _version switch
        {
            NearMissVersion.Version1 => SelectNearMiss1(x, majorityIndices, minorityIndices, targetCount),
            NearMissVersion.Version2 => SelectNearMiss2(x, majorityIndices, minorityIndices, targetCount),
            NearMissVersion.Version3 => SelectNearMiss3(x, majorityIndices, minorityIndices, targetCount),
            _ => SelectNearMiss1(x, majorityIndices, minorityIndices, targetCount)
        };
    }

    /// <summary>
    /// NearMiss-1: Keep majority samples closest to their nearest minority neighbors.
    /// </summary>
    private List<int> SelectNearMiss1(
        Matrix<T> x,
        List<int> majorityIndices,
        List<int> minorityIndices,
        int targetCount)
    {
        // For each majority sample, compute distance to nearest minority sample
        var majorityDistances = new List<(int index, double distance)>();

        foreach (int majIdx in majorityIndices)
        {
            var majSample = x.GetRow(majIdx);
            double minDistance = double.MaxValue;

            foreach (int minIdx in minorityIndices)
            {
                var minSample = x.GetRow(minIdx);
                double dist = NumOps.ToDouble(EuclideanDistance(majSample, minSample));
                if (dist < minDistance)
                {
                    minDistance = dist;
                }
            }

            majorityDistances.Add((majIdx, minDistance));
        }

        // Select majority samples with smallest distance to nearest minority
        return majorityDistances
            .OrderBy(d => d.distance)
            .Take(targetCount)
            .Select(d => d.index)
            .ToList();
    }

    /// <summary>
    /// NearMiss-2: Keep majority samples closest to their farthest minority neighbors.
    /// </summary>
    private List<int> SelectNearMiss2(
        Matrix<T> x,
        List<int> majorityIndices,
        List<int> minorityIndices,
        int targetCount)
    {
        // For each majority sample, compute distance to farthest minority sample
        var majorityDistances = new List<(int index, double distance)>();

        foreach (int majIdx in majorityIndices)
        {
            var majSample = x.GetRow(majIdx);
            double maxDistance = 0;

            foreach (int minIdx in minorityIndices)
            {
                var minSample = x.GetRow(minIdx);
                double dist = NumOps.ToDouble(EuclideanDistance(majSample, minSample));
                if (dist > maxDistance)
                {
                    maxDistance = dist;
                }
            }

            majorityDistances.Add((majIdx, maxDistance));
        }

        // Select majority samples with smallest distance to farthest minority
        return majorityDistances
            .OrderBy(d => d.distance)
            .Take(targetCount)
            .Select(d => d.index)
            .ToList();
    }

    /// <summary>
    /// NearMiss-3: Keep k nearest majority neighbors for each minority sample.
    /// </summary>
    private List<int> SelectNearMiss3(
        Matrix<T> x,
        List<int> majorityIndices,
        List<int> minorityIndices,
        int targetCount)
    {
        int effectiveK = Math.Min(_kNeighbors, majorityIndices.Count);
        var selectedIndices = new HashSet<int>();

        // For each minority sample, find k nearest majority neighbors
        foreach (int minIdx in minorityIndices)
        {
            var minSample = x.GetRow(minIdx);
            var distances = new List<(int index, double distance)>();

            foreach (int majIdx in majorityIndices)
            {
                var majSample = x.GetRow(majIdx);
                double dist = NumOps.ToDouble(EuclideanDistance(minSample, majSample));
                distances.Add((majIdx, dist));
            }

            // Add k nearest majority neighbors
            var nearest = distances
                .OrderBy(d => d.distance)
                .Take(effectiveK)
                .Select(d => d.index);

            foreach (int idx in nearest)
            {
                selectedIndices.Add(idx);
            }
        }

        // If we have more than target, take the ones closest overall
        if (selectedIndices.Count > targetCount)
        {
            // Rank by average distance to all minority samples
            var rankedIndices = selectedIndices
                .Select(majIdx =>
                {
                    var majSample = x.GetRow(majIdx);
                    double avgDist = minorityIndices
                        .Select(minIdx => NumOps.ToDouble(EuclideanDistance(majSample, x.GetRow(minIdx))))
                        .Average();
                    return (index: majIdx, avgDistance: avgDist);
                })
                .OrderBy(d => d.avgDistance)
                .Take(targetCount)
                .Select(d => d.index)
                .ToList();

            return rankedIndices;
        }

        // If we have fewer, just return what we have
        return selectedIndices.ToList();
    }
}

/// <summary>
/// Specifies the NearMiss version to use.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b>
/// - Version1: Keeps majority samples nearest to ANY minority sample
/// - Version2: Keeps majority samples nearest to the farthest minority sample
/// - Version3: Keeps k nearest majority neighbors for EACH minority sample
/// </para>
/// </remarks>
public enum NearMissVersion
{
    /// <summary>
    /// Select majority samples based on distance to nearest minority samples.
    /// </summary>
    Version1 = 1,

    /// <summary>
    /// Select majority samples based on distance to farthest minority samples.
    /// </summary>
    Version2 = 2,

    /// <summary>
    /// Select k nearest majority neighbors for each minority sample.
    /// </summary>
    Version3 = 3
}
