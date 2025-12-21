using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ActiveLearning.Batch;

/// <summary>
/// Simple ranked batch selection strategy with diversity filtering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This is the simplest batch selection strategy.
/// It ranks samples by their informativeness scores and selects the top-k,
/// optionally filtering out samples that are too similar to already-selected ones.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>Rank all candidates by informativeness score</description></item>
/// <item><description>Select top candidate</description></item>
/// <item><description>Filter remaining candidates by diversity threshold</description></item>
/// <item><description>Repeat until batch is complete</description></item>
/// </list>
///
/// <para><b>Trade-offs:</b></para>
/// <list type="bullet">
/// <item><description>Simple and fast</description></item>
/// <item><description>May miss globally diverse points</description></item>
/// <item><description>Works well when top candidates are naturally diverse</description></item>
/// </list>
/// </remarks>
public class RankedBatchStrategy<T, TInput, TOutput> : IBatchStrategy<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _minDiversityThreshold;
    private T _diversityTradeoff;

    /// <inheritdoc/>
    public string Name => "Ranked Batch Selection";

    /// <inheritdoc/>
    public T DiversityTradeoff
    {
        get => _diversityTradeoff;
        set => _diversityTradeoff = value;
    }

    /// <summary>
    /// Initializes a new RankedBatchStrategy with default settings.
    /// </summary>
    public RankedBatchStrategy()
        : this(minDiversityThreshold: 0.1, diversityTradeoff: 0.5)
    {
    }

    /// <summary>
    /// Initializes a new RankedBatchStrategy with specified parameters.
    /// </summary>
    /// <param name="minDiversityThreshold">Minimum diversity required between selected samples (0-1).</param>
    /// <param name="diversityTradeoff">Trade-off between informativeness and diversity (0-1).</param>
    public RankedBatchStrategy(double minDiversityThreshold, double diversityTradeoff = 0.5)
    {
        _minDiversityThreshold = NumOps.FromDouble(MathHelper.Clamp(minDiversityThreshold, 0.0, 1.0));
        _diversityTradeoff = NumOps.FromDouble(MathHelper.Clamp(diversityTradeoff, 0.0, 1.0));
    }

    /// <inheritdoc/>
    public int[] SelectBatch(
        int[] candidateIndices,
        Vector<T> scores,
        IDataset<T, TInput, TOutput> unlabeledPool,
        int batchSize)
    {
        if (candidateIndices.Length == 0)
        {
            return Array.Empty<int>();
        }

        int effectiveBatchSize = Math.Min(batchSize, candidateIndices.Length);

        // Create scored candidates
        var scoredCandidates = new List<(int Index, T Score)>();
        for (int i = 0; i < candidateIndices.Length; i++)
        {
            int poolIndex = candidateIndices[i];
            scoredCandidates.Add((poolIndex, scores[i]));
        }

        // Sort by score (descending)
        scoredCandidates.Sort((a, b) => NumOps.Compare(b.Score, a.Score));

        // Greedy selection with diversity filtering
        var selected = new List<int>();
        var selectedInputs = new List<TInput>();

        foreach (var (poolIndex, _) in scoredCandidates)
        {
            if (selected.Count >= effectiveBatchSize)
            {
                break;
            }

            var candidate = unlabeledPool.GetInput(poolIndex);

            // Check diversity with already selected samples
            bool isDiverse = true;
            foreach (var selectedInput in selectedInputs)
            {
                var diversity = ComputeDiversity(candidate, selectedInput);
                if (NumOps.Compare(diversity, _minDiversityThreshold) < 0)
                {
                    isDiverse = false;
                    break;
                }
            }

            if (isDiverse)
            {
                selected.Add(poolIndex);
                selectedInputs.Add(candidate);
            }
        }

        // If we didn't get enough diverse samples, fill with highest-scoring remaining
        if (selected.Count < effectiveBatchSize)
        {
            foreach (var (poolIndex, _) in scoredCandidates)
            {
                if (selected.Count >= effectiveBatchSize)
                {
                    break;
                }

                if (!selected.Contains(poolIndex))
                {
                    selected.Add(poolIndex);
                }
            }
        }

        return selected.ToArray();
    }

    /// <inheritdoc/>
    public T ComputeDiversity(TInput sample1, TInput sample2)
    {
        // Compute diversity based on distance between samples
        if (sample1 is Vector<T> vec1 && sample2 is Vector<T> vec2)
        {
            return ComputeNormalizedDistance(vec1, vec2);
        }

        if (sample1 is T[] arr1 && sample2 is T[] arr2)
        {
            return ComputeNormalizedDistance(new Vector<T>(arr1), new Vector<T>(arr2));
        }

        // Default: assume samples are diverse
        return NumOps.One;
    }

    #region Private Methods

    private T ComputeNormalizedDistance(Vector<T> a, Vector<T> b)
    {
        int length = Math.Min(a.Length, b.Length);
        if (length == 0)
        {
            return NumOps.One;
        }

        T sumSquared = NumOps.Zero;
        T maxPossible = NumOps.Zero;

        for (int i = 0; i < length; i++)
        {
            var diff = NumOps.Subtract(a[i], b[i]);
            sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(diff, diff));

            // Estimate max distance based on value magnitudes
            var absA = NumOps.Abs(a[i]);
            var absB = NumOps.Abs(b[i]);
            var maxVal = NumOps.Compare(absA, absB) >= 0 ? absA : absB;
            var maxDiff = NumOps.Multiply(NumOps.FromDouble(2), maxVal);
            maxPossible = NumOps.Add(maxPossible, NumOps.Multiply(maxDiff, maxDiff));
        }

        if (NumOps.Compare(maxPossible, NumOps.Zero) <= 0)
        {
            return NumOps.One;
        }

        var distance = NumOps.Sqrt(sumSquared);
        var maxDistance = NumOps.Sqrt(maxPossible);

        // Normalize to [0, 1]
        return NumOps.Divide(distance, NumOps.Add(maxDistance, NumOps.FromDouble(1e-10)));
    }

    #endregion
}
