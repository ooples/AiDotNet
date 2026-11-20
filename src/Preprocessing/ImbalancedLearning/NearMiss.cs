using AiDotNet.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
/// - For each majority sample, find the average distance to its k closest minority samples
/// - Keep majority samples with smallest such distances
/// - Focuses on boundary samples
///
/// NearMiss-2: Keep majority samples closest to farthest minority samples
/// - For each majority sample, find the average distance to its k farthest minority samples
/// - Keep majority samples with smallest such distances
/// - Keeps samples that are "in between" the minority class
///
/// NearMiss-3: Keep majority samples closest to each minority sample
/// - For each minority sample, find k nearest majority neighbors (the short list)
/// - Of the short list, keep the samples with the largest average distance to their k closest minority samples
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
/// - Mani &amp; Zhang (2003). "kNN Approach to Unbalanced Data Distributions"
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
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
    /// var trainX = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
    /// var trainY = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
    /// // Default: NearMiss-1 with balanced classes
    /// var nearmiss = new NearMiss&lt;double&gt;();
    ///
    /// // NearMiss-3 which keeps k nearest majority neighbors per minority sample
    /// var nearmiss2 = new NearMiss&lt;double&gt;(version: NearMissVersion.Version3, kNeighbors: 5);
    ///
    /// // Apply to data
    /// var (newX, newY) = nearmiss2.Resample(trainX, trainY);
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
    /// NearMiss-1 (Mani &amp; Zhang, 2003; imbalanced-learn): keeps the majority samples whose average distance
    /// to their k nearest minority samples is smallest.
    /// </summary>
    private List<int> SelectNearMiss1(
        Matrix<T> x,
        List<int> majorityIndices,
        List<int> minorityIndices,
        int targetCount)
    {
        var distances = MajorityToMinorityDistances(x, majorityIndices, minorityIndices);
        var scores = NeighborDistanceSums(distances, nearest: true);
        return SelectByScore(majorityIndices, scores, targetCount, largest: false);
    }

    /// <summary>
    /// NearMiss-2 (Mani &amp; Zhang, 2003; imbalanced-learn): keeps the majority samples whose average distance
    /// to their k farthest minority samples is smallest.
    /// </summary>
    private List<int> SelectNearMiss2(
        Matrix<T> x,
        List<int> majorityIndices,
        List<int> minorityIndices,
        int targetCount)
    {
        var distances = MajorityToMinorityDistances(x, majorityIndices, minorityIndices);
        var scores = NeighborDistanceSums(distances, nearest: false);
        return SelectByScore(majorityIndices, scores, targetCount, largest: false);
    }

    /// <summary>
    /// NearMiss-3 (Mani &amp; Zhang, 2003; imbalanced-learn): the k nearest majority samples of every minority
    /// sample are short-listed; of those, keeps the ones whose average distance to their k nearest minority
    /// samples is largest.
    /// </summary>
    private List<int> SelectNearMiss3(
        Matrix<T> x,
        List<int> majorityIndices,
        List<int> minorityIndices,
        int targetCount)
    {
        if (majorityIndices.Count == 0 || minorityIndices.Count == 0)
        {
            return new List<int>();
        }

        // [majority, minority] distances; column j holds the distances from minority sample j
        var distances = MajorityToMinorityDistances(x, majorityIndices, minorityIndices);
        int effectiveK = Math.Min(_kNeighbors, majorityIndices.Count);
        var minorityToMajority = Engine.TensorTranspose(distances);

        // Short-list (in ascending majority order, like numpy's unique) the k nearest majority samples of
        // each minority sample
        var shortListed = new SortedSet<int>(Enumerable.Range(0, minorityIndices.Count)
            .SelectMany(j => PreprocessingTensorOps.StableOrder(
                PreprocessingTensorOps.RowVector(minorityToMajority, j), descending: false).Take(effectiveK)));
        var candidates = shortListed.ToArray();

        var candidateDistances = Engine.TensorIndexSelect(
            distances, new Tensor<int>((int[])candidates.Clone(), new[] { candidates.Length }), 0);
        var scores = NeighborDistanceSums(candidateDistances, nearest: true);
        var candidateIndices = candidates.Select(c => majorityIndices[c]).ToList();
        return SelectByScore(candidateIndices, scores, targetCount, largest: true);
    }

    /// <summary>The engine used for the whole-array distance work.</summary>
    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>Euclidean distances [majority, minority] between the majority and the minority samples.</summary>
    private static Tensor<T> MajorityToMinorityDistances(Matrix<T> x, List<int> majorityIndices, List<int> minorityIndices)
    {
        var data = Tensor<T>.FromMatrix(x);
        var minority = minorityIndices.ToArray();
        var rows = majorityIndices
            .Select(majIdx => ResamplingTensorOps.Distances(data, majIdx, minority))
            .ToArray();
        if (rows.Length == 0)
        {
            return new Tensor<T>(new[] { 0, minority.Length });
        }

        return Engine.TensorStack(rows, 0);
    }

    /// <summary>
    /// For each row of a [rows, minority] distance tensor, the sum of its k smallest (or k largest) distances,
    /// k = min(n_neighbors, minority count). The sum ranks rows the same way as the average.
    /// </summary>
    private Tensor<T> NeighborDistanceSums(Tensor<T> distances, bool nearest)
    {
        int rows = distances.Shape[0];
        int minority = distances.Shape[1];
        int k = Math.Min(_kNeighbors, minority);
        if (rows == 0 || k == 0)
        {
            return new Tensor<T>(new[] { rows });
        }

        var sorted = Engine.TensorSort(distances, 1, false).Item1;
        var selected = Engine.TensorSlice(sorted, new[] { 0, nearest ? 0 : minority - k }, new[] { rows, k });
        return Engine.ReduceSum(selected, new[] { 1 }, false);
    }

    /// <summary>
    /// The <paramref name="targetCount"/> candidates with the smallest (or largest) scores; equal scores keep
    /// candidate order (Python's stable sort in imbalanced-learn).
    /// </summary>
    private static List<int> SelectByScore(List<int> candidates, Tensor<T> scores, int targetCount, bool largest)
    {
        return PreprocessingTensorOps.StableOrder(scores, descending: largest)
            .Take(targetCount)
            .Select(i => candidates[i])
            .ToList();
    }
}

/// <summary>
/// Specifies the NearMiss version to use.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b>
/// - Version1: Keeps majority samples nearest (on average) to their k closest minority samples
/// - Version2: Keeps majority samples nearest (on average) to their k farthest minority samples
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
