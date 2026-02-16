using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ScoreBased;

/// <summary>
/// Hill Climbing — greedy score-based DAG structure learning.
/// </summary>
/// <remarks>
/// <para>
/// Hill climbing searches over individual DAGs by greedily applying the single-edge operation
/// (add, remove, or reverse) that most improves the BIC score.
/// </para>
/// <para>
/// <b>For Beginners:</b> Hill climbing is the simplest score-based approach. At each step,
/// it tries adding, removing, or flipping every possible edge and picks the change that
/// improves the score the most.
/// </para>
/// <para>
/// Reference: Heckerman et al. (1995), "Learning Bayesian Networks".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class HillClimbingAlgorithm<T> : ScoreBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "Hill Climbing";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public HillClimbingAlgorithm(CausalDiscoveryOptions? options = null)
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

        for (int iteration = 0; iteration < MaxIterations; iteration++)
        {
            double bestImprovement = 0;
            int bestType = -1; // 0=add, 1=remove, 2=reverse
            int bestFrom = -1, bestTo = -1;

            // Try all add operations
            for (int to = 0; to < d; to++)
            {
                if (parentSets[to].Count >= MaxParents) continue;
                for (int from = 0; from < d; from++)
                {
                    if (from == to || parentSets[to].Contains(from)) continue;
                    if (WouldCreateCycle(parentSets, from, to, d)) continue;

                    var testParents = new HashSet<int>(parentSets[to]) { from };
                    double imp = ComputeBIC(data, to, testParents) - scores[to];
                    if (imp > bestImprovement)
                    {
                        bestImprovement = imp;
                        bestType = 0;
                        bestFrom = from;
                        bestTo = to;
                    }
                }
            }

            // Try all remove operations
            for (int to = 0; to < d; to++)
            {
                foreach (int from in parentSets[to])
                {
                    var testParents = new HashSet<int>(parentSets[to]);
                    testParents.Remove(from);
                    double imp = ComputeBIC(data, to, testParents) - scores[to];
                    if (imp > bestImprovement)
                    {
                        bestImprovement = imp;
                        bestType = 1;
                        bestFrom = from;
                        bestTo = to;
                    }
                }
            }

            // Try all reverse operations
            for (int to = 0; to < d; to++)
            {
                foreach (int from in parentSets[to])
                {
                    var testParentsTo = new HashSet<int>(parentSets[to]);
                    testParentsTo.Remove(from);

                    if (parentSets[from].Count >= MaxParents) continue;

                    var testParentsFrom = new HashSet<int>(parentSets[from]) { to };

                    var tempParents = (HashSet<int>[])parentSets.Clone();
                    tempParents[to] = testParentsTo;
                    tempParents[from] = testParentsFrom;
                    if (WouldCreateCycle(tempParents, to, from, d)) continue;

                    double impTo = ComputeBIC(data, to, testParentsTo) - scores[to];
                    double impFrom = ComputeBIC(data, from, testParentsFrom) - scores[from];
                    double totalImp = impTo + impFrom;

                    if (totalImp > bestImprovement)
                    {
                        bestImprovement = totalImp;
                        bestType = 2;
                        bestFrom = from;
                        bestTo = to;
                    }
                }
            }

            if (bestType < 0) break;

            switch (bestType)
            {
                case 0:
                    parentSets[bestTo].Add(bestFrom);
                    scores[bestTo] = ComputeBIC(data, bestTo, parentSets[bestTo]);
                    break;
                case 1:
                    parentSets[bestTo].Remove(bestFrom);
                    scores[bestTo] = ComputeBIC(data, bestTo, parentSets[bestTo]);
                    break;
                case 2:
                    parentSets[bestTo].Remove(bestFrom);
                    parentSets[bestFrom].Add(bestTo);
                    scores[bestTo] = ComputeBIC(data, bestTo, parentSets[bestTo]);
                    scores[bestFrom] = ComputeBIC(data, bestFrom, parentSets[bestFrom]);
                    break;
            }
        }

        // Build adjacency using Matrix<T>
        var W = new Matrix<T>(d, d);
        for (int to = 0; to < d; to++)
            foreach (int from in parentSets[to])
                W[from, to] = NumOps.FromDouble(Math.Max(0.01, ComputeAbsCorrelation(data, from, to)));

        return W;
    }
}
