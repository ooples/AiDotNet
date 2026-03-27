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
                // Use popcount to avoid allocating when set exceeds MaxParents
                int bitCount = BitCount(s);
                if (bitCount <= MaxParents)
                {
                    var parentsOfS = BitmaskToSet(s);
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
        ArrayPolyfill.Fill(dp, double.NegativeInfinity);
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

        // Build adjacency matrix with multivariate OLS-estimated edge weights
        // Each child is regressed against ALL of its selected parents jointly
        var W = new Matrix<T>(d, d);
        for (int v = 0; v < d; v++)
        {
            var parents = optimalParents[v].ToList();
            if (parents.Count == 0) continue;

            var coefficients = EstimateMultivariateOLS(data, n, parents, v);
            for (int p = 0; p < parents.Count; p++)
            {
                W[parents[p], v] = NumOps.FromDouble(coefficients[p]);
            }
        }

        return W;
    }

    /// <summary>
    /// Counts the number of set bits in an integer (population count).
    /// </summary>
    private static int BitCount(int n)
    {
        int count = 0;
        while (n != 0) { count++; n &= n - 1; }
        return count;
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
    /// Multivariate OLS: regress child on all parents jointly.
    /// Returns coefficient vector (one per parent).
    /// </summary>
    private double[] EstimateMultivariateOLS(Matrix<T> data, int n, List<int> parents, int child)
    {
        int p = parents.Count;
        if (p == 0) return [];

        // Compute means
        var means = new double[p + 1];
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < p; k++)
                means[k] += NumOps.ToDouble(data[i, parents[k]]);
            means[p] += NumOps.ToDouble(data[i, child]);
        }
        for (int k = 0; k <= p; k++)
            means[k] /= n;

        // Build X'X and X'y (normal equations)
        var XtX = new double[p, p];
        var Xty = new double[p];

        for (int i = 0; i < n; i++)
        {
            var dx = new double[p];
            for (int k = 0; k < p; k++)
                dx[k] = NumOps.ToDouble(data[i, parents[k]]) - means[k];
            double dy = NumOps.ToDouble(data[i, child]) - means[p];

            for (int a = 0; a < p; a++)
            {
                Xty[a] += dx[a] * dy;
                for (int b = a; b < p; b++)
                {
                    XtX[a, b] += dx[a] * dx[b];
                }
            }
        }

        // Symmetrize and add ridge
        for (int a = 0; a < p; a++)
        {
            XtX[a, a] += 1e-10;
            for (int b = a + 1; b < p; b++)
                XtX[b, a] = XtX[a, b];
        }

        // Solve via Cholesky-like approach (small p, so direct Gaussian elimination)
        var coeffs = SolveSmallSystem(XtX, Xty, p);
        return coeffs;
    }

    // SolveSmallSystem is inherited from CausalDiscoveryBase<T>
}
