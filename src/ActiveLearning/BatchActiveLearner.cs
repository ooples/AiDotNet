using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.ActiveLearning;

/// <summary>
/// Turns a per-sample informativeness (uncertainty) ranking into a diversity-aware batch selection,
/// so the chosen batch covers the pool instead of collecting near-duplicate uncertain samples.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Naive active learning takes the top-N most uncertain samples; those are frequently redundant (many
/// near-identical points from the same uncertain region), wasting labeling budget. This selects the
/// batch greedily by <c>informativeness − diversityWeight · redundancy</c>, where redundancy is a
/// sample's maximum similarity to what is already chosen (BADGE / facility-location style). It layers on
/// top of <i>any</i> uncertainty strategy — the strategy supplies the scores, this makes the batch
/// diverse.
/// </para>
/// </remarks>
public sealed class BatchActiveLearner<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly double _diversityWeight;

    /// <summary>Creates a batch active learner.</summary>
    /// <param name="diversityWeight">
    /// Redundancy penalty weight. 0 reduces to pure uncertainty sampling; higher values push the batch to
    /// cover more of the pool. Defaults to 0.5.
    /// </param>
    public BatchActiveLearner(double diversityWeight = 0.5)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _diversityWeight = diversityWeight;
    }

    /// <summary>
    /// Ranks the pool and selects a diversity-aware batch.
    /// </summary>
    /// <param name="informativeness">Per-sample uncertainty scores (length = pool size).</param>
    /// <param name="representation">
    /// Per-sample representation rows (pool size × features) used to measure redundancy — a model
    /// embedding when available, otherwise the raw input features.
    /// </param>
    /// <param name="batchSize">Number of samples to select.</param>
    /// <param name="strategyName">The configured strategy's name, for the report.</param>
    /// <param name="representationSpace">Label for the representation space used, for the report.</param>
    /// <returns>The ranked pool plus the selected batch.</returns>
    public ActiveLearningSelection<T> Select(
        Vector<T> informativeness,
        Matrix<T> representation,
        int batchSize,
        string strategyName,
        string representationSpace)
    {
        var engine = AiDotNetEngine.Current;
        int n = informativeness.Length;
        int budget = Math.Max(0, Math.Min(batchSize, n));
        if (n == 0)
        {
            return new ActiveLearningSelection<T>
            {
                BatchSize = budget,
                StrategyName = strategyName,
                RepresentationSpace = representationSpace,
                DiversityWeight = _diversityWeight,
            };
        }

        var scores = Tensor<T>.FromVector(informativeness).Reshape(new[] { 1, n });

        // Normalize informativeness to [0, 1] so the diversity weight is scale-independent.
        var norm = new Tensor<T>(new[] { 1, n });
        var min = engine.TensorMinValue(scores);
        var range = _numOps.Subtract(engine.TensorMaxValue(scores), min);
        if (_numOps.GreaterThan(range, _numOps.Zero))
        {
            norm = engine.TensorDivideScalar(engine.TensorSubtractScalar(scores, min), range);
        }

        // Representation rows and their L2 norms for cosine similarity.
        var rows = ActiveLearningEngineOps.AsRows(Tensor<T>.FromMatrix(representation), n, representation.Columns);
        var rowNorm = ActiveLearningEngineOps.RowNorms(rows).Reshape(new[] { 1, n });
        var hasNorm = engine.TensorGreaterThan(rowNorm, _numOps.Zero);

        var selected = new List<int>(budget);
        var selectionOrder = new int[n];
        for (int i = 0; i < n; i++) selectionOrder[i] = -1;
        var isSelected = new bool[n];
        var marginalGain = new Tensor<T>(new[] { 1, n });
        var maxSimToSelected = new Tensor<T>(new[] { 1, n }); // redundancy of each sample against the chosen set
        var weight = _numOps.FromDouble(_diversityWeight);

        for (int step = 0; step < budget; step++)
        {
            var adjusted = engine.TensorSubtract(norm, engine.TensorMultiplyScalar(maxSimToSelected, weight));
            int best = ActiveLearningEngineOps.ArgMaxAbove(adjusted, isSelected, _numOps.MinValue);
            if (best < 0) break;

            selectionOrder[best] = step;
            isSelected[best] = true;
            var chosen = new bool[n];
            chosen[best] = true;
            marginalGain = engine.TensorWhere(chosen, adjusted, marginalGain);
            selected.Add(best);

            // Update every remaining sample's redundancy against the newly chosen point.
            var similarity = CosineSimilarity(rows, rowNorm, hasNorm, best);
            maxSimToSelected = engine.TensorMax(maxSimToSelected, similarity);
        }

        // Unselected samples keep their raw normalized informativeness as marginal gain.
        marginalGain = engine.TensorWhere(isSelected, marginalGain, norm);

        // Selected batch first (in selection order), then the rest by informativeness.
        var byInformativeness = ActiveLearningEngineOps.StableOrder(Tensor<T>.FromVector(informativeness), descending: true);
        var order = new List<int>(n);
        order.AddRange(selected);
        foreach (var index in byInformativeness)
        {
            if (!isSelected[index]) order.Add(index);
        }

        var informativenessValues = informativeness.ToArray();
        var gains = marginalGain.ToArray();
        var ranking = new List<ActiveLearningCandidate<T>>(n);
        foreach (var index in order)
        {
            ranking.Add(new ActiveLearningCandidate<T>
            {
                PoolIndex = index,
                Informativeness = informativenessValues[index],
                SelectionOrder = selectionOrder[index],
                MarginalGain = gains[index],
            });
        }

        return new ActiveLearningSelection<T>
        {
            Ranking = ranking,
            SelectedIndices = selected.ToArray(),
            BatchSize = budget,
            StrategyName = strategyName,
            RepresentationSpace = representationSpace,
            DiversityWeight = _diversityWeight,
        };
    }

    /// <summary>
    /// Redundancy of every sample against sample <paramref name="k"/>: cosine similarity mapped to
    /// [0, 1] by clamping negatives (opposite directions are not redundant), 0 when either row has
    /// zero norm.
    /// </summary>
    /// <returns>A [1, n] tensor.</returns>
    private Tensor<T> CosineSimilarity(Tensor<T> rows, Tensor<T> rowNorm, Tensor<T> hasNorm, int k)
    {
        var engine = AiDotNetEngine.Current;
        int n = rows.Shape[0];
        var shape = new[] { 1, n };

        // Dot products of unit rows: a.b and |a| |b| overflow in Half
        var cos = ActiveLearningEngineOps.CosineBlock(rows, rowNorm, k, 1);

        var hasNormK = engine.TensorBroadcastTo(engine.TensorSlice(hasNorm, new[] { 0, k }, new[] { 1, 1 }), shape);
        var valid = engine.TensorMultiply(hasNorm, hasNormK);
        return engine.TensorWhere(valid, engine.TensorMax(cos, _numOps.Zero), new Tensor<T>(shape));
    }
}
