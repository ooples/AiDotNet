using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ScoreBased;

/// <summary>
/// GES (Greedy Equivalence Search) — score-based causal discovery over equivalence classes.
/// </summary>
/// <remarks>
/// <para>
/// GES searches over Markov equivalence classes of DAGs using two phases:
/// <list type="number">
/// <item><b>Forward phase:</b> Greedily adds edges that most improve the BIC score.</item>
/// <item><b>Backward phase:</b> Greedily removes edges that improve the BIC score.</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> GES builds a causal graph by first adding edges that improve the
/// model fit, then removing edges that are unnecessary. It uses a score (BIC) that rewards
/// fitting the data well while penalizing too many edges.
/// </para>
/// <para>
/// Reference: Chickering (2002), "Optimal Structure Identification with Greedy Search", JMLR.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GESAlgorithm<T> : ScoreBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "GES";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public GESAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyScoreOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int d = data.Columns;

        var parentSets = new HashSet<int>[d];
        for (int i = 0; i < d; i++) parentSets[i] = [];

        var scores = new double[d];
        for (int i = 0; i < d; i++)
            scores[i] = ComputeBIC(data, i, parentSets[i]);

        // Forward phase: add edges
        bool improved = true;
        int forwardIter = 0;
        while (improved && forwardIter < MaxIterations)
        {
            improved = false;
            forwardIter++;

            int bestFrom = -1, bestTo = -1;
            double bestImprovement = 0;

            for (int to = 0; to < d; to++)
            {
                if (parentSets[to].Count >= MaxParents) continue;

                for (int from = 0; from < d; from++)
                {
                    if (from == to || parentSets[to].Contains(from)) continue;
                    if (WouldCreateCycle(parentSets, from, to, d)) continue;

                    var testParents = new HashSet<int>(parentSets[to]) { from };
                    double newScore = ComputeBIC(data, to, testParents);
                    double improvement = newScore - scores[to];

                    if (improvement > bestImprovement)
                    {
                        bestImprovement = improvement;
                        bestFrom = from;
                        bestTo = to;
                    }
                }
            }

            if (bestFrom >= 0)
            {
                parentSets[bestTo].Add(bestFrom);
                scores[bestTo] = ComputeBIC(data, bestTo, parentSets[bestTo]);
                improved = true;
            }
        }

        // Backward phase: remove edges
        improved = true;
        int backwardIter = 0;
        while (improved && backwardIter < MaxIterations)
        {
            improved = false;
            backwardIter++;

            int bestFrom = -1, bestTo = -1;
            double bestImprovement = 0;

            for (int to = 0; to < d; to++)
            {
                foreach (int from in parentSets[to])
                {
                    var testParents = new HashSet<int>(parentSets[to]);
                    testParents.Remove(from);
                    double newScore = ComputeBIC(data, to, testParents);
                    double improvement = newScore - scores[to];

                    if (improvement > bestImprovement)
                    {
                        bestImprovement = improvement;
                        bestFrom = from;
                        bestTo = to;
                    }
                }
            }

            if (bestFrom >= 0)
            {
                parentSets[bestTo].Remove(bestFrom);
                scores[bestTo] = ComputeBIC(data, bestTo, parentSets[bestTo]);
                improved = true;
            }
        }

        // Build weighted adjacency using Matrix<T>
        var W = new Matrix<T>(d, d);
        for (int to = 0; to < d; to++)
        {
            foreach (int from in parentSets[to])
            {
                double weight = Math.Abs(ComputeRegCoef(data, from, to));
                W[from, to] = NumOps.FromDouble(Math.Max(weight, 0.01));
            }
        }

        return W;
    }

    private double ComputeRegCoef(Matrix<T> data, int from, int to)
    {
        int n = data.Rows;
        double meanX = 0, meanY = 0;
        for (int i = 0; i < n; i++)
        {
            meanX += NumOps.ToDouble(data[i, from]);
            meanY += NumOps.ToDouble(data[i, to]);
        }
        meanX /= n; meanY /= n;

        double sxy = 0, sxx = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = NumOps.ToDouble(data[i, from]) - meanX;
            sxy += dx * (NumOps.ToDouble(data[i, to]) - meanY);
            sxx += dx * dx;
        }

        return sxx > 1e-10 ? sxy / sxx : 0;
    }
}
