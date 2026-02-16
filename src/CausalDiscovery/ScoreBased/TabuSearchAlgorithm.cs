using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ScoreBased;

/// <summary>
/// Tabu Search — score-based DAG learning with memory to escape local optima.
/// </summary>
/// <remarks>
/// <para>
/// Tabu search extends hill climbing by maintaining a "tabu list" of recently visited
/// states that cannot be revisited.
/// </para>
/// <para>
/// <b>For Beginners:</b> Tabu search is like hill climbing but with a memory. If you just
/// climbed from point A to point B, you're not allowed to go back to A for a while.
/// </para>
/// <para>
/// Reference: Glover (1989, 1990), "Tabu Search".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabuSearchAlgorithm<T> : ScoreBasedBase<T>
{
    private const int DEFAULT_TABU_SIZE = 100;

    /// <inheritdoc/>
    public override string Name => "Tabu Search";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public TabuSearchAlgorithm(CausalDiscoveryOptions? options = null)
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

        double bestTotalScore = scores.Sum();
        var bestParentSets = CloneParentSets(parentSets, d);

        var tabuQueue = new Queue<(int type, int from, int to)>();
        var tabuSet = new HashSet<(int type, int from, int to)>();
        double currentTotal = scores.Sum();

        for (int iteration = 0; iteration < MaxIterations; iteration++)
        {
            double bestImprovement = double.NegativeInfinity;
            int bestType = -1, bestFrom = -1, bestTo = -1;

            for (int to = 0; to < d; to++)
            {
                // Add operations
                if (parentSets[to].Count < MaxParents)
                {
                    for (int from = 0; from < d; from++)
                    {
                        if (from == to || parentSets[to].Contains(from)) continue;
                        if (WouldCreateCycle(parentSets, from, to)) continue;

                        bool isTabu = tabuSet.Contains((1, from, to));

                        var testParents = new HashSet<int>(parentSets[to]) { from };
                        double imp = ComputeBIC(data, to, testParents) - scores[to];

                        double newTotal = currentTotal + imp;
                        if (isTabu && newTotal <= bestTotalScore) continue;

                        if (imp > bestImprovement)
                        {
                            bestImprovement = imp;
                            bestType = 0;
                            bestFrom = from;
                            bestTo = to;
                        }
                    }
                }

                // Remove operations
                foreach (int from in parentSets[to])
                {
                    bool isTabu = tabuSet.Contains((0, from, to));

                    var testParents = new HashSet<int>(parentSets[to]);
                    testParents.Remove(from);
                    double imp = ComputeBIC(data, to, testParents) - scores[to];

                    double newTotal = currentTotal + imp;
                    if (isTabu && newTotal <= bestTotalScore) continue;

                    if (imp > bestImprovement)
                    {
                        bestImprovement = imp;
                        bestType = 1;
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
                    var entry1 = (1, bestFrom, bestTo);
                    tabuQueue.Enqueue(entry1);
                    tabuSet.Add(entry1);
                    break;
                case 1:
                    parentSets[bestTo].Remove(bestFrom);
                    scores[bestTo] = ComputeBIC(data, bestTo, parentSets[bestTo]);
                    var entry2 = (0, bestFrom, bestTo);
                    tabuQueue.Enqueue(entry2);
                    tabuSet.Add(entry2);
                    break;
            }

            while (tabuQueue.Count > DEFAULT_TABU_SIZE)
                tabuSet.Remove(tabuQueue.Dequeue());

            currentTotal = scores.Sum();
            if (currentTotal > bestTotalScore)
            {
                bestTotalScore = currentTotal;
                bestParentSets = CloneParentSets(parentSets, d);
            }
        }

        // Build adjacency from best parent sets using Matrix<T>
        var W = new Matrix<T>(d, d);
        for (int to = 0; to < d; to++)
            foreach (int from in bestParentSets[to])
                W[from, to] = NumOps.FromDouble(Math.Max(1e-10, ComputeAbsCorrelation(data, from, to)));

        return W;
    }

    private static HashSet<int>[] CloneParentSets(HashSet<int>[] parentSets, int d)
    {
        var clone = new HashSet<int>[d];
        for (int i = 0; i < d; i++)
            clone[i] = new HashSet<int>(parentSets[i]);
        return clone;
    }
}
