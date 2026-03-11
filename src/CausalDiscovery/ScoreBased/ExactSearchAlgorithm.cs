using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ScoreBased;

/// <summary>
/// Exact Search (Dynamic Programming) — optimal DAG structure learning.
/// </summary>
/// <remarks>
/// <para>
/// Uses dynamic programming over subsets of variables to find the globally optimal
/// DAG structure according to a decomposable score (BIC). The algorithm works in two phases:
/// </para>
/// <para>
/// <b>Algorithm (Silander-Myllymaki):</b>
/// <list type="number">
/// <item>Phase 1 (Parent scoring): For each variable and each subset of potential parents,
///   compute the BIC score. Store the best parent set for each variable given each ancestor set.</item>
/// <item>Phase 2 (Order search): Use DP over variable orderings.
///   For each subset S, compute the best DAG score by trying each variable as the "last" in
///   the ordering. The best parents for that variable come from S minus that variable.</item>
/// <item>Backtrack through the DP table to reconstruct the optimal parent assignments.</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> This algorithm finds the absolute best causal graph by
/// systematically checking all possibilities using clever math shortcuts (dynamic
/// programming). It's guaranteed to find the optimal solution but only works for
/// small datasets (up to about 20 variables due to O(2^d) complexity).
/// </para>
/// <para>
/// Reference: Silander and Myllymaki (2006), "A Simple Approach for Finding the
/// Globally Optimal Bayesian Network Structure".
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
[ModelPaper("A Simple Approach for Finding the Globally Optimal Bayesian Network Structure", "https://arxiv.org/abs/1206.6875", Year = 2006, Authors = "Tomi Silander, Petri Myllymaki")]
public class ExactSearchAlgorithm<T> : ScoreBasedBase<T>
{
    private const int MaxVariablesForExactSearch = 20;

    /// <inheritdoc/>
    public override string Name => "Exact Search (DP)";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes Exact Search with optional configuration.
    /// </summary>
    public ExactSearchAlgorithm(CausalDiscoveryOptions? options = null) { ApplyScoreOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n < 2 || d < 2) return new Matrix<T>(d, d);

        // Cap at MaxVariablesForExactSearch to avoid memory explosion
        if (d > MaxVariablesForExactSearch)
        {
            throw new InvalidOperationException(
                $"Exact search requires O(2^d) memory. {d} variables exceeds the maximum of {MaxVariablesForExactSearch}. " +
                "Use GES or FGES for larger datasets.");
        }

        int totalSubsets = 1 << d; // 2^d subsets

        // Phase 1: Compute best parent set for each variable given each candidate set
        // bestLocalScore[v, S] = best BIC score for variable v with parents ⊆ S
        // bestLocalParents[v, S] = the parent set achieving that score
        var bestLocalScore = new double[d, totalSubsets];
        var bestLocalParents = new int[d, totalSubsets]; // bitmask of parents

        for (int v = 0; v < d; v++)
        {
            // Initialize: no parents
            bestLocalScore[v, 0] = ComputeBIC(data, v, []);
            bestLocalParents[v, 0] = 0;

            // For each subset S (excluding v), find best parent set ⊆ S
            for (int s = 1; s < totalSubsets; s++)
            {
                // Skip subsets containing v (can't be own parent)
                if ((s & (1 << v)) != 0)
                {
                    bestLocalScore[v, s] = bestLocalScore[v, s & ~(1 << v)];
                    bestLocalParents[v, s] = bestLocalParents[v, s & ~(1 << v)];
                    continue;
                }

                // Try removing each element from S; best is either:
                // (a) best of S without that element, or
                // (b) best with that element added as parent
                bestLocalScore[v, s] = double.NegativeInfinity;

                // The subset S with all elements as parents
                var parentsOfS = BitmaskToSet(s);
                if (parentsOfS.Count <= MaxParents)
                {
                    double score = ComputeBIC(data, v, parentsOfS);
                    if (score > bestLocalScore[v, s])
                    {
                        bestLocalScore[v, s] = score;
                        bestLocalParents[v, s] = s;
                    }
                }

                // Also check all subsets of S by removing one element at a time
                for (int bit = 0; bit < d; bit++)
                {
                    if ((s & (1 << bit)) == 0) continue;
                    int subS = s & ~(1 << bit);
                    if (bestLocalScore[v, subS] > bestLocalScore[v, s])
                    {
                        bestLocalScore[v, s] = bestLocalScore[v, subS];
                        bestLocalParents[v, s] = bestLocalParents[v, subS];
                    }
                }
            }
        }

        // Phase 2: DP over orderings
        // dp[S] = best total BIC score achievable using variables in S
        // lastInOrder[S] = which variable was last in the optimal ordering for S
        var dp = new double[totalSubsets];
        var lastInOrder = new int[totalSubsets];
        Array.Fill(dp, double.NegativeInfinity);
        dp[0] = 0;

        for (int s = 1; s < totalSubsets; s++)
        {
            // Try each variable in S as the "last" variable in the ordering
            for (int v = 0; v < d; v++)
            {
                if ((s & (1 << v)) == 0) continue; // v not in S

                int predecessors = s & ~(1 << v); // S \ {v}
                double candidateScore = dp[predecessors] + bestLocalScore[v, predecessors];

                if (candidateScore > dp[s])
                {
                    dp[s] = candidateScore;
                    lastInOrder[s] = v;
                }
            }
        }

        // Backtrack to find optimal parent sets
        var optimalParents = new HashSet<int>[d];
        for (int v = 0; v < d; v++)
            optimalParents[v] = [];

        int remaining = totalSubsets - 1; // all variables
        while (remaining != 0)
        {
            int v = lastInOrder[remaining];
            int predecessors = remaining & ~(1 << v);
            int parentBitmask = bestLocalParents[v, predecessors];
            optimalParents[v] = BitmaskToSet(parentBitmask);
            remaining = predecessors;
        }

        // Build adjacency matrix with OLS-estimated edge weights
        var W = new Matrix<T>(d, d);
        for (int v = 0; v < d; v++)
        {
            foreach (int parent in optimalParents[v])
            {
                double weight = EstimateEdgeWeight(data, n, parent, v);
                W[parent, v] = NumOps.FromDouble(weight);
            }
        }

        return W;
    }

    /// <summary>
    /// Converts a bitmask to a HashSet of variable indices.
    /// </summary>
    private static HashSet<int> BitmaskToSet(int mask)
    {
        var set = new HashSet<int>();
        for (int i = 0; mask != 0; i++, mask >>= 1)
        {
            if ((mask & 1) != 0)
                set.Add(i);
        }
        return set;
    }

    /// <summary>
    /// Estimates the OLS regression coefficient for parent→child edge weight.
    /// </summary>
    private double EstimateEdgeWeight(Matrix<T> data, int n, int parent, int child)
    {
        double meanP = 0, meanC = 0;
        for (int i = 0; i < n; i++)
        {
            meanP += NumOps.ToDouble(data[i, parent]);
            meanC += NumOps.ToDouble(data[i, child]);
        }
        meanP /= n;
        meanC /= n;

        double cov = 0, varP = 0;
        for (int i = 0; i < n; i++)
        {
            double dp = NumOps.ToDouble(data[i, parent]) - meanP;
            cov += dp * (NumOps.ToDouble(data[i, child]) - meanC);
            varP += dp * dp;
        }

        return varP > 1e-10 ? cov / varP : 0;
    }
}
