using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ScoreBased;

/// <summary>
/// GRaSP (Greedy Relaxation of Sparsest Permutation) — permutation-based causal discovery.
/// </summary>
/// <remarks>
/// <para>
/// GRaSP searches over permutations (orderings) of variables and selects the sparsest
/// DAG consistent with each ordering. It uses greedy local moves (adjacent transpositions)
/// to explore the permutation space, accepting moves that reduce the total number of edges.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize with an ordering (e.g., based on marginal variance)</item>
/// <item>For the current ordering, compute the optimal parent set for each variable
///   (parents must precede the variable in the ordering)</item>
/// <item>Count total edges (sparsity measure)</item>
/// <item>Try all adjacent transpositions (swap positions i and i+1)</item>
/// <item>For each swap, recompute parent sets for the two affected variables</item>
/// <item>Accept the swap that yields the largest reduction in total edges (or best BIC if tied)</item>
/// <item>Repeat until no swap improves sparsity</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> GRaSP tries different orderings of variables and for each ordering,
/// finds the simplest (sparsest) causal graph. It's designed to find graphs with fewer
/// edges, which often corresponds to the true causal structure.
/// </para>
/// <para>
/// Reference: Lam et al. (2022), "Greedy Relaxations of the Sparsest Permutation Algorithm".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("Greedy Relaxations of the Sparsest Permutation Algorithm", "https://arxiv.org/abs/2211.02763", Year = 2022, Authors = "Wai-Yin Lam, Bryan Andrews, Joseph Ramsey")]
public class GRaSPAlgorithm<T> : ScoreBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "GRaSP";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes GRaSP with optional configuration.
    /// </summary>
    public GRaSPAlgorithm(CausalDiscoveryOptions? options = null) { ApplyScoreOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n < 2 || d < 2) return new Matrix<T>(d, d);

        // Initialize ordering using marginal variance (low variance = likely exogenous)
        var ordering = InitializeOrdering(data, d);

        // Compute initial parent sets and edge count
        var parentSets = ComputeOptimalParents(data, ordering);
        int currentEdges = CountEdges(parentSets, d);
        double currentScore = ComputeTotalScore(data, parentSets, d);

        // Greedy adjacent transpositions
        bool improved = true;
        int iteration = 0;

        while (improved && iteration < MaxIterations)
        {
            improved = false;
            iteration++;

            int bestSwapPos = -1;
            int bestEdgeDelta = 0;
            double bestScoreDelta = double.NegativeInfinity;

            for (int pos = 0; pos < d - 1; pos++)
            {
                // Try swapping ordering[pos] and ordering[pos+1]
                int v1 = ordering[pos];
                int v2 = ordering[pos + 1];

                // Swap
                ordering[pos] = v2;
                ordering[pos + 1] = v1;

                // Recompute parent sets for affected variables
                var newParents = ComputeOptimalParents(data, ordering);
                int newEdges = CountEdges(newParents, d);
                double newScore = ComputeTotalScore(data, newParents, d);

                int edgeDelta = newEdges - currentEdges;
                double scoreDelta = newScore - currentScore;

                // Prefer fewer edges; break ties by BIC score
                if (edgeDelta < bestEdgeDelta ||
                    (edgeDelta == bestEdgeDelta && scoreDelta > bestScoreDelta))
                {
                    bestSwapPos = pos;
                    bestEdgeDelta = edgeDelta;
                    bestScoreDelta = scoreDelta;
                }

                // Swap back
                ordering[pos] = v1;
                ordering[pos + 1] = v2;
            }

            if (bestSwapPos >= 0 && (bestEdgeDelta < 0 || (bestEdgeDelta == 0 && bestScoreDelta > 0)))
            {
                // Apply the best swap
                int v1 = ordering[bestSwapPos];
                int v2 = ordering[bestSwapPos + 1];
                ordering[bestSwapPos] = v2;
                ordering[bestSwapPos + 1] = v1;

                parentSets = ComputeOptimalParents(data, ordering);
                currentEdges = CountEdges(parentSets, d);
                currentScore = ComputeTotalScore(data, parentSets, d);
                improved = true;
            }
        }

        // Build adjacency matrix
        return BuildAdjacencyMatrix(data, parentSets, d);
    }

    /// <summary>
    /// Initializes ordering based on marginal variance (ascending — low variance first).
    /// </summary>
    private int[] InitializeOrdering(Matrix<T> data, int d)
    {
        int n = data.Rows;
        var variances = new double[d];

        for (int j = 0; j < d; j++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++) mean += NumOps.ToDouble(data[i, j]);
            mean /= n;

            double variance = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - mean;
                variance += diff * diff;
            }
            variances[j] = variance / n;
        }

        var ordering = Enumerable.Range(0, d).ToArray();
        Array.Sort(ordering, (a, b) => variances[a].CompareTo(variances[b]));
        return ordering;
    }

    /// <summary>
    /// For each variable in the ordering, finds the best parent set from predecessors using BIC.
    /// </summary>
    private HashSet<int>[] ComputeOptimalParents(Matrix<T> data, int[] ordering)
    {
        int d = ordering.Length;
        var parentSets = new HashSet<int>[d];
        for (int i = 0; i < d; i++)
            parentSets[i] = [];

        for (int idx = 1; idx < d; idx++)
        {
            int target = ordering[idx];
            var bestParents = new HashSet<int>();
            double bestScore = ComputeBIC(data, target, bestParents);

            // Greedily add predecessors
            bool addImproved = true;
            while (addImproved && bestParents.Count < MaxParents)
            {
                addImproved = false;
                int bestCandidate = -1;
                double bestCandidateScore = bestScore;

                for (int predIdx = 0; predIdx < idx; predIdx++)
                {
                    int candidate = ordering[predIdx];
                    if (bestParents.Contains(candidate)) continue;

                    var testParents = new HashSet<int>(bestParents) { candidate };
                    double score = ComputeBIC(data, target, testParents);

                    if (score > bestCandidateScore)
                    {
                        bestCandidateScore = score;
                        bestCandidate = candidate;
                    }
                }

                if (bestCandidate >= 0)
                {
                    bestParents.Add(bestCandidate);
                    bestScore = bestCandidateScore;
                    addImproved = true;
                }
            }

            parentSets[target] = bestParents;
        }

        return parentSets;
    }

    private static int CountEdges(HashSet<int>[] parentSets, int d)
    {
        int count = 0;
        for (int i = 0; i < d; i++)
            count += parentSets[i].Count;
        return count;
    }

    private double ComputeTotalScore(Matrix<T> data, HashSet<int>[] parentSets, int d)
    {
        double total = 0;
        for (int i = 0; i < d; i++)
            total += ComputeBIC(data, i, parentSets[i]);
        return total;
    }

    private Matrix<T> BuildAdjacencyMatrix(Matrix<T> data, HashSet<int>[] parentSets, int d)
    {
        int n = data.Rows;
        var W = new Matrix<T>(d, d);

        for (int child = 0; child < d; child++)
        {
            foreach (int parent in parentSets[child])
            {
                double meanP = 0, meanC = 0;
                for (int i = 0; i < n; i++)
                {
                    meanP += NumOps.ToDouble(data[i, parent]);
                    meanC += NumOps.ToDouble(data[i, child]);
                }
                meanP /= n; meanC /= n;

                double cov = 0, varP = 0;
                for (int i = 0; i < n; i++)
                {
                    double dp = NumOps.ToDouble(data[i, parent]) - meanP;
                    cov += dp * (NumOps.ToDouble(data[i, child]) - meanC);
                    varP += dp * dp;
                }

                W[parent, child] = NumOps.FromDouble(varP > 1e-10 ? cov / varP : 0);
            }
        }

        return W;
    }
}
