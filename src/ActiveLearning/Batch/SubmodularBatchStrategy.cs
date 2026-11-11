using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// Citation URL corrected. arXiv 1505.05123 is "Reed-Muller Codes Achieve Capacity on Erasure
// Channels" — an unrelated coding-theory paper. This work appeared at ICML 2015 (PMLR v37) and is not
// on arXiv, so the canonical proceedings URL replaces the invented preprint id.
[ResearchPaper("Submodularity in Data Subset Selection and Active Learning",
    "https://proceedings.mlr.press/v37/wei15.html",
    Year = 2015,
    Authors = "Kai Wei, Rishabh Iyer, Jeff Bilmes")]
[ComponentType(ComponentType.ActiveLearner)]
[PipelineStage(PipelineStage.Training)]
public class SubmodularBatchStrategy<T, TInput, TOutput> : ISubmodularBatchStrategy<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static readonly T NegativeGamma = NumOps.Negate(NumOps.FromDouble(0.1)); // RBF kernel width

    private readonly SubmodularObjective _objective;
    private readonly T _lambda; // Informativeness weight
    private T _diversityTradeoff;
    [Scratch]
    private Tensor<T>? _cachedFeatures;

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
        var uniformScores = AiDotNetEngine.Current.Fill(candidateIndices.Length, NumOps.One);

        return GreedyMaximizationWithScores(candidateIndices, uniformScores, batchSize);
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

    private Tensor<T> ExtractFeatures(int[] indices, IDataset<T, TInput, TOutput> pool)
    {
        var features = new List<Vector<T>>(indices.Length);

        foreach (var idx in indices)
        {
            var input = pool.GetInput(idx);
            var feature = ConvertToVector(input);
            features.Add(feature);
        }

        return features.Count == 0
            ? new Tensor<T>(new[] { 0, 0 })
            : ActiveLearningEngineOps.StackVectors(features);
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

    /// <summary>
    /// Greedy maximization of λ·informativeness + (1-λ)·marginal diversity gain. Every remaining
    /// candidate's exact marginal gain is evaluated each round (the previous lazy upper-bound skip
    /// only pruned candidates whose exact gain could not win, so the selected set is the same for
    /// non-negative weights). Ties go to the lowest candidate index.
    /// </summary>
    private int[] GreedyMaximizationWithScores(int[] candidateIndices, Vector<T> scores, int batchSize)
    {
        var features = _cachedFeatures;
        int n = candidateIndices.Length;
        if (features is null || n == 0 || features.Shape[0] < n)
        {
            return Array.Empty<int>();
        }

        var engine = AiDotNetEngine.Current;
        int dim = features.Shape[1];
        var rows = features.Shape[0] == n
            ? features
            : engine.TensorSlice(features, new[] { 0, 0 }, new[] { n, dim });

        var informativeness = engine.TensorMultiplyScalar(Tensor<T>.FromVector(scores.GetSubVector(0, n)), _lambda);
        var selected = new List<int>();
        var isSelected = new bool[n];

        // Running per-sample state of the selected set
        var maxSimToSelected = new Tensor<T>(new[] { n });
        var minDistToSelected = new Tensor<T>(new[] { n });
        minDistToSelected.Fill(NumOps.MaxValue);

        for (int b = 0; b < batchSize && selected.Count < n; b++)
        {
            var diversityGain = _objective switch
            {
                SubmodularObjective.GraphCut => GraphCutGains(rows, isSelected),
                SubmodularObjective.LogDeterminant => selected.Count == 0 ? Ones(n) : minDistToSelected,
                _ => FacilityLocationGains(rows, maxSimToSelected)
            };

            var totalGain = engine.TensorAdd(informativeness, engine.TensorMultiplyScalar(diversityGain, _diversityTradeoff));
            int bestIdx = ActiveLearningEngineOps.ArgMaxAbove(totalGain.Reshape(new[] { 1, n }), isSelected, NumOps.MinValue);
            if (bestIdx < 0)
            {
                break;
            }

            selected.Add(bestIdx);
            isSelected[bestIdx] = true;

            var newest = engine.TensorSlice(rows, new[] { bestIdx, 0 }, new[] { 1, dim });
            maxSimToSelected = engine.TensorMax(maxSimToSelected, SimilaritiesToRows(rows, newest).Reshape(new[] { n }));
            minDistToSelected = engine.TensorMin(minDistToSelected, ActiveLearningEngineOps.DistancesToPoint(rows, newest));
        }

        // Map back to pool indices
        return selected.Select(i => candidateIndices[i]).ToArray();
    }

    private T ComputeMarginalGainInternal(IReadOnlyList<int> currentSelection, int candidateIndex)
    {
        var features = _cachedFeatures;
        if (features is null || candidateIndex < 0 || candidateIndex >= features.Shape[0])
        {
            return NumOps.Zero;
        }

        var engine = AiDotNetEngine.Current;
        int n = features.Shape[0];
        int dim = features.Shape[1];
        var candidate = engine.TensorSlice(features, new[] { candidateIndex, 0 }, new[] { 1, dim });
        var selection = currentSelection.ToArray();
        var selectionRows = selection.Length == 0
            ? new Tensor<T>(new[] { 0, dim })
            : engine.TensorIndexSelect(features, new Tensor<int>(selection, new[] { selection.Length }), 0);
        var self = new bool[n];
        self[candidateIndex] = true;

        switch (_objective)
        {
            case SubmodularObjective.GraphCut:
            {
                // Σ_{u≠v} ±sim(u, v): reward unselected points, penalize selected ones
                var isSelected = new bool[n];
                foreach (var s in selection)
                {
                    isSelected[s] = true;
                }

                var signed = engine.TensorMultiply(SimilaritiesToRows(features, candidate).Reshape(new[] { n }), Signs(isSelected));
                return engine.TensorSum(engine.TensorMaskedFill(signed, self, NumOps.Zero));
            }

            case SubmodularObjective.LogDeterminant:
            {
                // Approximation: minimum distance to the selected set (first selection gets 1)
                if (selection.Length == 0)
                {
                    return NumOps.One;
                }

                return engine.TensorMinValue(ActiveLearningEngineOps.DistancesToPoint(selectionRows, candidate));
            }

            default:
            {
                // Facility location: Δf(v|S) = Σ_{u≠v} max(0, sim(u,v) - max(0, max_{s∈S} sim(u,s)))
                var maxSimToSelected = selection.Length == 0
                    ? new Tensor<T>(new[] { n })
                    : engine.TensorMax(engine.ReduceMax(SimilaritiesToRows(features, selectionRows), new[] { 1 }, keepDims: false), NumOps.Zero);
                var improvement = PositivePart(engine.TensorSubtract(SimilaritiesToRows(features, candidate).Reshape(new[] { n }), maxSimToSelected));
                return engine.TensorSum(engine.TensorMaskedFill(improvement, self, NumOps.Zero));
            }
        }
    }

    /// <summary>
    /// Facility-location gains of every candidate v: Σ_{u≠v} max(0, sim(u,v) - m(u)), where m(u) is
    /// the (non-negative) maximum similarity of u to the selected set.
    /// </summary>
    private Tensor<T> FacilityLocationGains(Tensor<T> rows, Tensor<T> maxSimToSelected)
    {
        var engine = AiDotNetEngine.Current;
        return SimilarityColumnSums(rows, (sim, start, count) =>
        {
            var shape = new[] { count, rows.Shape[0] };
            var m = engine.TensorSlice(maxSimToSelected, new[] { start }, new[] { count }).Reshape(new[] { count, 1 });
            return PositivePart(engine.TensorSubtract(sim, engine.TensorBroadcastTo(m, shape)));
        });
    }

    /// <summary>
    /// Graph-cut gains of every candidate v: Σ_{u≠v, u∉S} sim(u,v) - Σ_{u≠v, u∈S} sim(u,v).
    /// </summary>
    private Tensor<T> GraphCutGains(Tensor<T> rows, bool[] isSelected)
    {
        var engine = AiDotNetEngine.Current;
        var signs = Signs(isSelected);
        return SimilarityColumnSums(rows, (sim, start, count) =>
        {
            var shape = new[] { count, rows.Shape[0] };
            var s = engine.TensorSlice(signs, new[] { start }, new[] { count }).Reshape(new[] { count, 1 });
            return engine.TensorMultiply(sim, engine.TensorBroadcastTo(s, shape));
        });
    }

    /// <summary>
    /// Σ_{u≠v} f(sim(u, v)) for every column v, computed blockwise over the rows u.
    /// </summary>
    private static Tensor<T> SimilarityColumnSums(Tensor<T> rows, Func<Tensor<T>, int, int, Tensor<T>> transform)
    {
        var engine = AiDotNetEngine.Current;
        int n = rows.Shape[0];
        int dim = rows.Shape[1];
        int blockRows = ActiveLearningEngineOps.PairwiseBlockRows(n, dim);
        var sums = new Tensor<T>(new[] { n });

        for (int start = 0; start < n; start += blockRows)
        {
            int count = Math.Min(blockRows, n - start);
            var block = engine.TensorSlice(rows, new[] { start, 0 }, new[] { count, dim });
            var terms = transform(SimilaritiesToRows(block, rows), start, count);
            var offDiagonal = engine.TensorMaskedFill(terms, ActiveLearningEngineOps.SelfMask(start, count, n), NumOps.Zero);
            sums = engine.TensorAdd(sums, engine.ReduceSum(offDiagonal, new[] { 0 }, keepDims: false));
        }

        return sums;
    }

    /// <summary>
    /// RBF similarities exp(-γ·||a_i - b_j||²) between the rows of a [m, d] and a [k, d] tensor.
    /// </summary>
    /// <returns>A [m, k] tensor.</returns>
    private static Tensor<T> SimilaritiesToRows(Tensor<T> a, Tensor<T> b)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorExp(engine.TensorMultiplyScalar(ActiveLearningEngineOps.SquaredDistances(a, b), NegativeGamma));
    }

    /// <summary>
    /// Elementwise max(0, x), mapping NaN to zero like the scalar "x &gt; 0" test.
    /// </summary>
    private static Tensor<T> PositivePart(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorWhere(engine.TensorGreaterThan(values, NumOps.Zero), values, new Tensor<T>(values.Shape.ToArray()));
    }

    /// <summary>
    /// -1 for selected samples and +1 for the others.
    /// </summary>
    private static Tensor<T> Signs(bool[] isSelected)
    {
        var shape = new[] { isSelected.Length };
        var minusOne = new Tensor<T>(shape);
        minusOne.Fill(NumOps.Negate(NumOps.One));
        return AiDotNetEngine.Current.TensorWhere(isSelected, minusOne, Ones(isSelected.Length));
    }

    private static Tensor<T> Ones(int n)
    {
        var ones = new Tensor<T>(new[] { n });
        ones.Fill(NumOps.One);
        return ones;
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
        var engine = AiDotNetEngine.Current;
        var diff = engine.Subtract(a, b);
        return engine.DotProduct(diff, diff);
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
