using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Tabular.Undersampling;

/// <summary>
/// Implements NearMiss undersampling for imbalanced datasets.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> NearMiss is a smarter undersampling method that selects majority
/// class samples based on their distance to minority samples. Unlike random undersampling, it
/// considers the spatial relationships between classes.</para>
///
/// <para><b>NearMiss Variants:</b>
/// <list type="bullet">
/// <item><b>NearMiss-1:</b> Selects majority samples whose average distance to k nearest
///   minority samples is smallest. These are majority samples close to the minority class.</item>
/// <item><b>NearMiss-2:</b> Selects majority samples whose average distance to k farthest
///   minority samples is smallest. These are majority samples close to all minority samples.</item>
/// <item><b>NearMiss-3:</b> For each minority sample, keeps the k nearest majority samples.
///   This surrounds each minority sample with selected majority samples.</item>
/// </list>
/// </para>
///
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>NearMiss-1: When you want to keep majority samples near the decision boundary</item>
/// <item>NearMiss-2: When minority class has outliers you want to handle</item>
/// <item>NearMiss-3: When you want to preserve local structure around minority samples</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Zhang & Mani, "kNN Approach to Unbalanced Data Distributions" (2003)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class NearMissUnderSampler<T> : IUnderSampler<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// NearMiss variant versions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each version uses a different strategy for selecting
    /// which majority samples to keep.</para>
    /// </remarks>
    public enum NearMissVersion
    {
        /// <summary>Keep majority samples close to nearest minority samples.</summary>
        NearMiss1,
        /// <summary>Keep majority samples close to farthest minority samples.</summary>
        NearMiss2,
        /// <summary>Keep k majority samples nearest to each minority sample.</summary>
        NearMiss3
    }

    /// <summary>
    /// Gets the NearMiss version to use.
    /// </summary>
    /// <remarks>
    /// <para>Default: NearMiss1</para>
    /// </remarks>
    public NearMissVersion Version { get; }

    /// <summary>
    /// Gets the number of nearest neighbors to consider.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For NearMiss-1 and NearMiss-2, this is the number of minority
    /// neighbors used to calculate average distance. For NearMiss-3, this is the number of
    /// majority samples to keep around each minority sample.</para>
    /// <para>Default: 3</para>
    /// </remarks>
    public int NNeighbors { get; }

    /// <summary>
    /// Gets the target ratio between minority and majority samples after undersampling.
    /// </summary>
    /// <remarks>
    /// <para>Default: 1.0 (balanced classes)</para>
    /// </remarks>
    public double SamplingRatio { get; }

    /// <summary>
    /// Creates a new NearMiss undersampler.
    /// </summary>
    /// <param name="version">The NearMiss version to use (default: NearMiss1).</param>
    /// <param name="nNeighbors">Number of neighbors to consider (default: 3).</param>
    /// <param name="samplingRatio">Target ratio of minority to majority (default: 1.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Start with NearMiss-1 and default parameters. Try NearMiss-3
    /// if you need to preserve more local structure.</para>
    /// </remarks>
    public NearMissUnderSampler(
        NearMissVersion version = NearMissVersion.NearMiss1,
        int nNeighbors = 3,
        double samplingRatio = 1.0)
    {
        if (nNeighbors < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nNeighbors), "Number of neighbors must be at least 1.");
        }

        if (samplingRatio <= 0 || samplingRatio > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(samplingRatio),
                "Sampling ratio must be in range (0, 1].");
        }

        _numOps = MathHelper.GetNumericOperations<T>();
        Version = version;
        NNeighbors = nNeighbors;
        SamplingRatio = samplingRatio;
    }

    /// <summary>
    /// Performs NearMiss undersampling on the majority class.
    /// </summary>
    /// <param name="data">The full dataset.</param>
    /// <param name="labels">Class labels for each sample.</param>
    /// <param name="minorityLabel">The label value for the minority class.</param>
    /// <returns>Tuple of (undersampled data, undersampled labels).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method selects majority samples based on their
    /// distance relationship to minority samples, keeping samples that are most relevant
    /// for learning the decision boundary.</para>
    /// </remarks>
    public (Matrix<T> Data, Vector<T> Labels) Undersample(
        Matrix<T> data,
        Vector<T> labels,
        T minorityLabel)
    {
        int rows = data.Rows;
        int cols = data.Columns;

        // Separate minority and majority
        var minorityIndices = new List<int>();
        var majorityIndices = new List<int>();

        double minorityLabelVal = _numOps.ToDouble(minorityLabel);

        for (int i = 0; i < rows; i++)
        {
            if (_numOps.ToDouble(labels[i]).Equals(minorityLabelVal))
            {
                minorityIndices.Add(i);
            }
            else
            {
                majorityIndices.Add(i);
            }
        }

        if (minorityIndices.Count == 0 || majorityIndices.Count == 0)
        {
            return (data.Clone(), labels.Clone());
        }

        // Calculate target number of majority samples
        int targetMajority = (int)Math.Ceiling(minorityIndices.Count / SamplingRatio);

        if (majorityIndices.Count <= targetMajority)
        {
            return (data.Clone(), labels.Clone());
        }

        // Select majority samples based on version
        var selectedMajorityIndices = Version switch
        {
            NearMissVersion.NearMiss1 => SelectNearMiss1(data, minorityIndices, majorityIndices, targetMajority),
            NearMissVersion.NearMiss2 => SelectNearMiss2(data, minorityIndices, majorityIndices, targetMajority),
            NearMissVersion.NearMiss3 => SelectNearMiss3(data, minorityIndices, majorityIndices, targetMajority),
            _ => throw new InvalidOperationException($"Unknown NearMiss version: {Version}")
        };

        // Combine selected majority with all minority
        int totalSamples = minorityIndices.Count + selectedMajorityIndices.Count;
        var resultData = new Matrix<T>(totalSamples, cols);
        var resultLabels = new Vector<T>(totalSamples);

        int idx = 0;

        foreach (int minIdx in minorityIndices)
        {
            for (int c = 0; c < cols; c++)
            {
                resultData[idx, c] = data[minIdx, c];
            }
            resultLabels[idx] = labels[minIdx];
            idx++;
        }

        foreach (int majIdx in selectedMajorityIndices)
        {
            for (int c = 0; c < cols; c++)
            {
                resultData[idx, c] = data[majIdx, c];
            }
            resultLabels[idx] = labels[majIdx];
            idx++;
        }

        return (resultData, resultLabels);
    }

    /// <summary>
    /// NearMiss-1: Select majority samples closest to k nearest minority samples.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For each majority sample, find its k nearest minority
    /// neighbors and compute the average distance. Keep majority samples with smallest
    /// average distance (closest to minority class).</para>
    /// </remarks>
    private List<int> SelectNearMiss1(
        Matrix<T> data,
        List<int> minorityIndices,
        List<int> majorityIndices,
        int targetCount)
    {
        int cols = data.Columns;
        int effectiveK = Math.Min(NNeighbors, minorityIndices.Count);

        var majorityScores = new List<(int Index, double AvgDistance)>();

        foreach (int majIdx in majorityIndices)
        {
            // Find k nearest minority samples
            var distances = minorityIndices
                .Select(minIdx => (Index: minIdx, Distance: ComputeDistance(data, majIdx, minIdx, cols)))
                .OrderBy(x => x.Distance)
                .Take(effectiveK)
                .ToList();

            double avgDist = distances.Average(x => x.Distance);
            majorityScores.Add((majIdx, avgDist));
        }

        // Select majority samples with smallest average distance
        return majorityScores
            .OrderBy(x => x.AvgDistance)
            .Take(targetCount)
            .Select(x => x.Index)
            .ToList();
    }

    /// <summary>
    /// NearMiss-2: Select majority samples closest to k farthest minority samples.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For each majority sample, find its k farthest minority
    /// neighbors and compute the average distance. Keep majority samples with smallest
    /// average distance (closest to even the farthest minority samples).</para>
    /// </remarks>
    private List<int> SelectNearMiss2(
        Matrix<T> data,
        List<int> minorityIndices,
        List<int> majorityIndices,
        int targetCount)
    {
        int cols = data.Columns;
        int effectiveK = Math.Min(NNeighbors, minorityIndices.Count);

        var majorityScores = new List<(int Index, double AvgDistance)>();

        foreach (int majIdx in majorityIndices)
        {
            // Find k farthest minority samples
            var distances = minorityIndices
                .Select(minIdx => (Index: minIdx, Distance: ComputeDistance(data, majIdx, minIdx, cols)))
                .OrderByDescending(x => x.Distance)
                .Take(effectiveK)
                .ToList();

            double avgDist = distances.Average(x => x.Distance);
            majorityScores.Add((majIdx, avgDist));
        }

        // Select majority samples with smallest average distance to farthest minority
        return majorityScores
            .OrderBy(x => x.AvgDistance)
            .Take(targetCount)
            .Select(x => x.Index)
            .ToList();
    }

    /// <summary>
    /// NearMiss-3: For each minority sample, keep k nearest majority samples.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This version surrounds each minority sample with its
    /// k nearest majority neighbors. This preserves local structure but may select
    /// more or fewer than targetCount samples.</para>
    /// </remarks>
    private List<int> SelectNearMiss3(
        Matrix<T> data,
        List<int> minorityIndices,
        List<int> majorityIndices,
        int targetCount)
    {
        int cols = data.Columns;
        int effectiveK = Math.Min(NNeighbors, majorityIndices.Count);

        var selectedSet = new HashSet<int>();

        foreach (int minIdx in minorityIndices)
        {
            // Find k nearest majority samples
            var nearest = majorityIndices
                .Select(majIdx => (Index: majIdx, Distance: ComputeDistance(data, minIdx, majIdx, cols)))
                .OrderBy(x => x.Distance)
                .Take(effectiveK);

            foreach (var item in nearest)
            {
                selectedSet.Add(item.Index);
            }
        }

        // If we have too many, randomly subsample
        var selectedList = selectedSet.ToList();
        if (selectedList.Count > targetCount)
        {
            var rand = RandomHelper.CreateSecureRandom();
            selectedList = selectedList
                .OrderBy(_ => rand.Next())
                .Take(targetCount)
                .ToList();
        }

        return selectedList;
    }

    /// <summary>
    /// Computes Euclidean distance between two samples.
    /// </summary>
    private double ComputeDistance(Matrix<T> data, int idx1, int idx2, int cols)
    {
        double dist = 0;
        for (int c = 0; c < cols; c++)
        {
            double diff = _numOps.ToDouble(data[idx1, c]) - _numOps.ToDouble(data[idx2, c]);
            dist += diff * diff;
        }
        return Math.Sqrt(dist);
    }
}
