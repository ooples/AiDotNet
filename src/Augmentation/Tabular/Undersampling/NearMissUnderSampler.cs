using AiDotNet.Helpers;
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
/// <para><b>Reference:</b> Zhang &amp; Mani, "kNN Approach to Unbalanced Data Distributions" (2003)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class NearMissUnderSampler<T> : IUnderSampler<T>
{

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
        var minorityIndices = TabularResamplingHelper<T>.IndicesOfLabel(labels, minorityLabel);

        // imbalanced-learn handles every non-minority class on its own; with two classes this is the
        // single majority class.
        var classIndices = TabularResamplingHelper<T>.OtherLabels(labels, minorityLabel)
            .Select(label => TabularResamplingHelper<T>.IndicesOfLabel(labels, label))
            .ToList();

        if (minorityIndices.Length == 0 || classIndices.Count == 0)
        {
            return (data.Clone(), labels.Clone());
        }

        // Calculate target number of majority samples
        int targetMajority = (int)Math.Ceiling(minorityIndices.Length / SamplingRatio);

        if (classIndices.All(indices => indices.Length <= targetMajority))
        {
            return (data.Clone(), labels.Clone());
        }

        var minorityRows = TabularResamplingHelper<T>.Rows(data, minorityIndices);

        // All minority samples first, then each class's selection.
        var selected = new List<int>(minorityIndices);
        foreach (var majorityIndices in classIndices)
        {
            if (majorityIndices.Length <= targetMajority)
            {
                selected.AddRange(majorityIndices);
                continue;
            }

            selected.AddRange(Version switch
            {
                NearMissVersion.NearMiss1 => SelectByMinorityDistance(data, minorityRows, majorityIndices, targetMajority, farthestNeighbors: false, keepFarthest: false),
                NearMissVersion.NearMiss2 => SelectByMinorityDistance(data, minorityRows, majorityIndices, targetMajority, farthestNeighbors: true, keepFarthest: false),
                NearMissVersion.NearMiss3 => SelectNearMiss3(data, minorityRows, majorityIndices, targetMajority),
                _ => throw new InvalidOperationException($"Unknown NearMiss version: {Version}")
            });
        }

        return TabularResamplingHelper<T>.Select(data, labels, selected.ToArray());
    }

    /// <summary>
    /// Ranks the candidate majority samples by their mean distance to their k nearest (NearMiss-1, NearMiss-3)
    /// or k farthest (NearMiss-2) minority samples and keeps <paramref name="targetCount"/> of them.
    /// </summary>
    /// <remarks>
    /// NearMiss-1 and NearMiss-2 keep the smallest means, NearMiss-3 the largest (imbalanced-learn's
    /// "nearest" and "farthest" selection). Equal means keep their original order.
    /// </remarks>
    private List<int> SelectByMinorityDistance(
        Matrix<T> data,
        Tensor<T> minorityRows,
        int[] candidates,
        int targetCount,
        bool farthestNeighbors,
        bool keepFarthest)
    {
        int k = Math.Min(NNeighbors, minorityRows.Shape[0]);
        var means = new List<T>(candidates.Length);
        foreach (int candidate in candidates)
        {
            var distances = TabularResamplingHelper<T>.DistancesTo(minorityRows, data.GetRow(candidate).AsTensor());
            means.Add(TabularResamplingHelper<T>.MeanOfExtremes(distances, k, farthestNeighbors));
        }

        var order = TabularResamplingHelper<T>.StableOrder(new Vector<T>(means).AsTensor(), keepFarthest);
        return order.Take(targetCount).Select(position => candidates[position]).ToList();
    }

    /// <summary>
    /// NearMiss-3 (Mani and Zhang 2003; imbalanced-learn): keep each minority sample's k nearest majority
    /// samples as candidates, then keep the candidates whose mean distance to their k nearest minority
    /// samples is largest.
    /// </summary>
    private List<int> SelectNearMiss3(
        Matrix<T> data,
        Tensor<T> minorityRows,
        int[] majorityIndices,
        int targetCount)
    {
        int k = Math.Min(NNeighbors, majorityIndices.Length);
        var majorityRows = TabularResamplingHelper<T>.Rows(data, majorityIndices);
        var candidatePositions = new SortedSet<int>();
        int minorityCount = minorityRows.Shape[0];
        var engine = AiDotNetEngine.Current;
        for (int m = 0; m < minorityCount; m++)
        {
            var row = engine.TensorSlice(minorityRows, new[] { m, 0 }, new[] { 1, minorityRows.Shape[1] });
            var distances = TabularResamplingHelper<T>.DistancesTo(majorityRows, row);
            candidatePositions.UnionWith(TabularResamplingHelper<T>.StableOrder(distances, descending: false).Take(k));
        }

        var candidates = candidatePositions.Select(position => majorityIndices[position]).ToArray();
        return SelectByMinorityDistance(data, minorityRows, candidates, targetCount, farthestNeighbors: false, keepFarthest: true);
    }
}
