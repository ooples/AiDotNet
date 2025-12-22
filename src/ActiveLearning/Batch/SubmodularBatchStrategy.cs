using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ActiveLearning.Batch;

/// <summary>
/// Submodular batch selection strategy using facility location objectives.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Submodular functions have a special property called
/// "diminishing returns" - adding more similar samples provides less and less benefit.
/// This naturally encourages selecting diverse, representative samples.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>Start with an empty selection set</description></item>
/// <item><description>Greedily add samples that maximize marginal gain</description></item>
/// <item><description>The submodular property ensures diverse selections</description></item>
/// </list>
///
/// <para><b>Facility Location Objective:</b></para>
/// <para>F(S) = Σ max_{s∈S} sim(x, s) for all x in the data</para>
/// <para>This objective ensures selected samples cover the entire data space well.</para>
///
/// <para><b>Advantages:</b></para>
/// <list type="bullet">
/// <item><description>Theoretical guarantees: greedy achieves (1-1/e) optimal</description></item>
/// <item><description>Naturally balances diversity and coverage</description></item>
/// <item><description>Computationally efficient with lazy evaluation</description></item>
/// </list>
///
/// <para><b>Reference:</b> Wei et al. "Submodularity in Data Subset Selection and Active Learning" (ICML 2015)</para>
/// </remarks>
public class SubmodularBatchStrategy<T, TInput, TOutput> : ISubmodularBatchStrategy<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly SubmodularObjective _objective;
    private readonly T _lambda; // Informativeness weight
    private T _diversityTradeoff;
    private List<Vector<T>>? _cachedFeatures;

    /// <inheritdoc/>
    public string Name => $"Submodular Batch Selection ({_objective})";

    /// <inheritdoc/>
    public T DiversityTradeoff
    {
        get => _diversityTradeoff;
        set => _diversityTradeoff = value;
    }

    /// <summary>
    /// Initializes a new SubmodularBatchStrategy with default settings.
    /// </summary>
    public SubmodularBatchStrategy()
        : this(SubmodularObjective.FacilityLocation, lambda: 0.5)
    {
    }

    /// <summary>
    /// Initializes a new SubmodularBatchStrategy with specified parameters.
    /// </summary>
    /// <param name="objective">The submodular objective function to use.</param>
    /// <param name="lambda">Weight for informativeness term (0-1).</param>
    public SubmodularBatchStrategy(SubmodularObjective objective, double lambda = 0.5)
    {
        _objective = objective;
        _lambda = NumOps.FromDouble(MathHelper.Clamp(lambda, 0.0, 1.0));
        _diversityTradeoff = NumOps.FromDouble(1.0 - lambda);
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

        // Extract and cache features
        _cachedFeatures = ExtractFeatures(candidateIndices, unlabeledPool);

        // Run greedy submodular maximization with informativeness
        return GreedyMaximizationWithScores(candidateIndices, scores, effectiveBatchSize);
    }

    /// <inheritdoc/>
    public T ComputeMarginalGain(
        IReadOnlyList<int> currentSelection,
        int candidateIndex,
        IDataset<T, TInput, TOutput> unlabeledPool)
    {
        if (_cachedFeatures == null)
        {
            var indices = unlabeledPool.GetIndices();
            _cachedFeatures = ExtractFeatures(indices, unlabeledPool);
        }

        return ComputeMarginalGainInternal(currentSelection, candidateIndex);
    }

    /// <inheritdoc/>
    public int[] GreedyMaximization(
        int[] candidateIndices,
        IDataset<T, TInput, TOutput> unlabeledPool,
        int batchSize)
    {
        _cachedFeatures = ExtractFeatures(candidateIndices, unlabeledPool);

        // Use uniform scores for pure diversity selection
        var uniformScores = new T[candidateIndices.Length];
        for (int i = 0; i < candidateIndices.Length; i++)
        {
            uniformScores[i] = NumOps.One;
        }

        return GreedyMaximizationWithScores(candidateIndices, new Vector<T>(uniformScores), batchSize);
    }

    /// <inheritdoc/>
    public T ComputeDiversity(TInput sample1, TInput sample2)
    {
        if (sample1 is Vector<T> vec1 && sample2 is Vector<T> vec2)
        {
            // Diversity = 1 - similarity
            var similarity = ComputeSimilarity(vec1, vec2);
            return NumOps.Subtract(NumOps.One, similarity);
        }

        return NumOps.One;
    }

    #region Private Methods

    private List<Vector<T>> ExtractFeatures(int[] indices, IDataset<T, TInput, TOutput> pool)
    {
        var features = new List<Vector<T>>();

        foreach (var idx in indices)
        {
            var input = pool.GetInput(idx);
            var feature = ConvertToVector(input);
            features.Add(feature);
        }

        return features;
    }

    private Vector<T> ConvertToVector(TInput input)
    {
        if (input is Vector<T> vec)
        {
            return vec;
        }

        if (input is T[] arr)
        {
            return new Vector<T>(arr);
        }

        if (input is IReadOnlyList<T> list)
        {
            return new Vector<T>(list.ToArray());
        }

        if (input is T val)
        {
            return new Vector<T>(new[] { val });
        }

        return new Vector<T>(new[] { NumOps.Zero });
    }

    private int[] GreedyMaximizationWithScores(int[] candidateIndices, Vector<T> scores, int batchSize)
    {
        var selected = new List<int>();
        var remaining = new HashSet<int>(Enumerable.Range(0, candidateIndices.Length));

        // Lazy evaluation: cache upper bounds for efficiency
        var upperBounds = new T[candidateIndices.Length];
        for (int i = 0; i < candidateIndices.Length; i++)
        {
            upperBounds[i] = NumOps.MaxValue;
        }

        for (int b = 0; b < batchSize && remaining.Count > 0; b++)
        {
            T maxGain = NumOps.MinValue;
            int bestIdx = -1;

            // Find sample with maximum marginal gain
            foreach (var idx in remaining)
            {
                // Lazy evaluation: skip if upper bound is worse than current best
                if (bestIdx >= 0 && NumOps.Compare(upperBounds[idx], maxGain) < 0)
                {
                    continue;
                }

                // Compute marginal gain
                var diversityGain = ComputeMarginalGainInternal(selected, idx);

                // Combine with informativeness score
                var informativenessGain = scores[idx];
                var totalGain = ComputeCombinedGain(diversityGain, informativenessGain);

                upperBounds[idx] = totalGain;

                if (NumOps.Compare(totalGain, maxGain) > 0)
                {
                    maxGain = totalGain;
                    bestIdx = idx;
                }
            }

            if (bestIdx >= 0)
            {
                selected.Add(bestIdx);
                remaining.Remove(bestIdx);
            }
        }

        // Map back to pool indices
        return selected.Select(i => candidateIndices[i]).ToArray();
    }

    private T ComputeMarginalGainInternal(IReadOnlyList<int> currentSelection, int candidateIndex)
    {
        if (_cachedFeatures == null || candidateIndex >= _cachedFeatures.Count)
        {
            return NumOps.Zero;
        }

        return _objective switch
        {
            SubmodularObjective.FacilityLocation => FacilityLocationGain(currentSelection, candidateIndex),
            SubmodularObjective.GraphCut => GraphCutGain(currentSelection, candidateIndex),
            SubmodularObjective.LogDeterminant => LogDeterminantGain(currentSelection, candidateIndex),
            _ => FacilityLocationGain(currentSelection, candidateIndex)
        };
    }

    private T FacilityLocationGain(IReadOnlyList<int> currentSelection, int candidateIndex)
    {
        // Facility location: gain is improvement in max similarity for each point
        // Δf(v|S) = Σ max(0, sim(u,v) - max_{s∈S} sim(u,s))
        if (_cachedFeatures == null)
        {
            return NumOps.Zero;
        }

        T totalGain = NumOps.Zero;
        var candidateFeature = _cachedFeatures[candidateIndex];

        for (int u = 0; u < _cachedFeatures.Count; u++)
        {
            if (u == candidateIndex)
            {
                continue;
            }

            // Current max similarity to selected set
            T maxSimToSelected = NumOps.Zero;
            foreach (var s in currentSelection)
            {
                var sim = ComputeSimilarity(_cachedFeatures[u], _cachedFeatures[s]);
                if (NumOps.Compare(sim, maxSimToSelected) > 0)
                {
                    maxSimToSelected = sim;
                }
            }

            // Similarity to candidate
            var simToCandidate = ComputeSimilarity(_cachedFeatures[u], candidateFeature);

            // Marginal improvement
            var improvement = NumOps.Subtract(simToCandidate, maxSimToSelected);
            if (NumOps.Compare(improvement, NumOps.Zero) > 0)
            {
                totalGain = NumOps.Add(totalGain, improvement);
            }
        }

        return totalGain;
    }

    private T GraphCutGain(IReadOnlyList<int> currentSelection, int candidateIndex)
    {
        // Graph cut: gain is sum of similarities to unselected minus selected
        if (_cachedFeatures == null)
        {
            return NumOps.Zero;
        }

        T gain = NumOps.Zero;
        var candidateFeature = _cachedFeatures[candidateIndex];

        for (int u = 0; u < _cachedFeatures.Count; u++)
        {
            if (u == candidateIndex)
            {
                continue;
            }

            var sim = ComputeSimilarity(_cachedFeatures[u], candidateFeature);

            if (currentSelection.Contains(u))
            {
                // Penalty for similarity to already selected
                gain = NumOps.Subtract(gain, sim);
            }
            else
            {
                // Reward for covering unselected points
                gain = NumOps.Add(gain, sim);
            }
        }

        return gain;
    }

    private T LogDeterminantGain(IReadOnlyList<int> currentSelection, int candidateIndex)
    {
        // Log-determinant is computationally expensive; use approximation
        // Based on distance to nearest selected point
        if (_cachedFeatures == null)
        {
            return NumOps.Zero;
        }

        var candidateFeature = _cachedFeatures[candidateIndex];

        if (currentSelection.Count == 0)
        {
            // First selection: high gain
            return NumOps.One;
        }

        // Gain proportional to minimum distance to selected set
        T minDist = NumOps.MaxValue;
        foreach (var s in currentSelection)
        {
            var dist = ComputeDistance(candidateFeature, _cachedFeatures[s]);
            if (NumOps.Compare(dist, minDist) < 0)
            {
                minDist = dist;
            }
        }

        return minDist;
    }

    private T ComputeCombinedGain(T diversityGain, T informativenessGain)
    {
        // Linear combination: λ * informativeness + (1-λ) * diversity
        var infTerm = NumOps.Multiply(_lambda, informativenessGain);
        var divTerm = NumOps.Multiply(_diversityTradeoff, diversityGain);
        return NumOps.Add(infTerm, divTerm);
    }

    private T ComputeSimilarity(Vector<T> a, Vector<T> b)
    {
        // RBF (Gaussian) similarity
        var squaredDist = ComputeSquaredDistance(a, b);
        var gamma = NumOps.FromDouble(0.1); // Kernel width
        var exponent = NumOps.Negate(NumOps.Multiply(gamma, squaredDist));
        return NumOps.Exp(exponent);
    }

    private T ComputeSquaredDistance(Vector<T> a, Vector<T> b)
    {
        int length = Math.Min(a.Length, b.Length);
        T sum = NumOps.Zero;

        for (int i = 0; i < length; i++)
        {
            var diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return sum;
    }

    private T ComputeDistance(Vector<T> a, Vector<T> b)
    {
        return NumOps.Sqrt(ComputeSquaredDistance(a, b));
    }

    #endregion
}

/// <summary>
/// Types of submodular objective functions.
/// </summary>
public enum SubmodularObjective
{
    /// <summary>
    /// Facility location: maximize coverage of the data space.
    /// F(S) = Σ max_{s∈S} sim(x, s)
    /// </summary>
    FacilityLocation,

    /// <summary>
    /// Graph cut: balance diversity and representativeness.
    /// F(S) = Σ sim(S, V\S) - Σ sim(S, S)
    /// </summary>
    GraphCut,

    /// <summary>
    /// Log-determinant: maximize volume of selected points.
    /// F(S) = log det(I + α K_S) where K is kernel matrix
    /// </summary>
    LogDeterminant
}
