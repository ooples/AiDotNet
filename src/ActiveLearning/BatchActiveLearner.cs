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
    public ActiveLearningSelection Select(
        Vector<T> informativeness,
        Matrix<T> representation,
        int batchSize,
        string strategyName,
        string representationSpace)
    {
        int n = informativeness.Length;
        int budget = Math.Max(0, Math.Min(batchSize, n));

        // Normalize informativeness to [0, 1] so the diversity weight is scale-independent.
        var scores = new double[n];
        double min = double.PositiveInfinity, max = double.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            scores[i] = _numOps.ToDouble(informativeness[i]);
            if (scores[i] < min) min = scores[i];
            if (scores[i] > max) max = scores[i];
        }

        double range = max - min;
        var norm = new double[n];
        for (int i = 0; i < n; i++)
        {
            norm[i] = range > 0 ? (scores[i] - min) / range : 0.0;
        }

        // Precompute L2 norms of representation rows for cosine similarity.
        var rowNorm = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < representation.Columns; j++)
            {
                double v = _numOps.ToDouble(representation[i, j]);
                sum += v * v;
            }

            rowNorm[i] = Math.Sqrt(sum);
        }

        var selected = new List<int>(budget);
        var selectionOrder = new int[n];
        var marginalGain = new double[n];
        for (int i = 0; i < n; i++) selectionOrder[i] = -1;
        var maxSimToSelected = new double[n]; // redundancy of each sample against the chosen set

        for (int step = 0; step < budget; step++)
        {
            int best = -1;
            double bestAdjusted = double.NegativeInfinity;
            for (int i = 0; i < n; i++)
            {
                if (selectionOrder[i] >= 0) continue;
                double adjusted = norm[i] - (_diversityWeight * maxSimToSelected[i]);
                if (adjusted > bestAdjusted)
                {
                    bestAdjusted = adjusted;
                    best = i;
                }
            }

            if (best < 0) break;

            selectionOrder[best] = step;
            marginalGain[best] = bestAdjusted;
            selected.Add(best);

            // Update every remaining sample's redundancy against the newly chosen point.
            for (int i = 0; i < n; i++)
            {
                if (selectionOrder[i] >= 0) continue;
                double sim = CosineSimilarity(representation, rowNorm, i, best);
                if (sim > maxSimToSelected[i]) maxSimToSelected[i] = sim;
            }
        }

        // Unselected samples keep their raw normalized informativeness as marginal gain.
        for (int i = 0; i < n; i++)
        {
            if (selectionOrder[i] < 0) marginalGain[i] = norm[i];
        }

        var ranking = new List<ActiveLearningCandidate>(n);
        for (int i = 0; i < n; i++)
        {
            ranking.Add(new ActiveLearningCandidate
            {
                PoolIndex = i,
                Informativeness = scores[i],
                SelectionOrder = selectionOrder[i],
                MarginalGain = marginalGain[i],
            });
        }

        // Selected batch first (in selection order), then the rest by informativeness.
        ranking.Sort((a, b) =>
        {
            bool aSel = a.SelectionOrder >= 0, bSel = b.SelectionOrder >= 0;
            if (aSel && bSel) return a.SelectionOrder.CompareTo(b.SelectionOrder);
            if (aSel) return -1;
            if (bSel) return 1;
            return b.Informativeness.CompareTo(a.Informativeness);
        });

        return new ActiveLearningSelection
        {
            Ranking = ranking,
            SelectedIndices = selected.ToArray(),
            BatchSize = budget,
            StrategyName = strategyName,
            RepresentationSpace = representationSpace,
            DiversityWeight = _diversityWeight,
        };
    }

    private double CosineSimilarity(Matrix<T> rep, double[] rowNorm, int i, int k)
    {
        if (rowNorm[i] <= 0 || rowNorm[k] <= 0) return 0.0;
        double dot = 0;
        for (int j = 0; j < rep.Columns; j++)
        {
            dot += _numOps.ToDouble(rep[i, j]) * _numOps.ToDouble(rep[k, j]);
        }

        double cos = dot / (rowNorm[i] * rowNorm[k]);
        // Map cosine [-1, 1] to a [0, 1] redundancy: opposite directions are not redundant.
        return Math.Max(0.0, cos);
    }
}
