using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ScoreBased;

/// <summary>
/// K2 Algorithm — score-based learning with known variable ordering.
/// </summary>
/// <remarks>
/// <para>
/// K2 learns a Bayesian network structure given a known topological ordering of variables.
/// For each variable in order, it greedily adds parents from variables earlier in the ordering
/// that maximize the BIC score, up to a maximum number of parents.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Determine a variable ordering (using correlation-based heuristic if not provided)</item>
/// <item>For each variable X_i in order:</item>
/// <item>  Initialize parents(X_i) = empty</item>
/// <item>  Repeat: find the predecessor z that maximizes BIC(X_i | parents + z)</item>
/// <item>  Add z if BIC improves and |parents| &lt; maxParents</item>
/// <item>  Stop when no improvement or max parents reached</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> If you already know the rough order in which variables cause
/// each other (e.g., age before income before spending), K2 efficiently finds the
/// exact connections. It's very fast but requires this ordering as input. When no
/// ordering is provided, a heuristic based on correlation structure is used.
/// </para>
/// <para>
/// Reference: Cooper and Herskovits (1992), "A Bayesian Method for the Induction
/// of Probabilistic Networks from Data".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("A Bayesian Method for the Induction of Probabilistic Networks from Data", "https://link.springer.com/article/10.1007/BF00994110", Year = 1992, Authors = "Gregory F. Cooper, Edward Herskovits")]
public class K2Algorithm<T> : ScoreBasedBase<T>
{
    private readonly int _maxParentsPerNode;

    /// <inheritdoc/>
    public override string Name => "K2";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes K2 with optional configuration.
    /// </summary>
    public K2Algorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyScoreOptions(options);
        _maxParentsPerNode = options?.MaxParents ?? 4;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n < 2 || d < 2) return new Matrix<T>(d, d);

        // Determine variable ordering using correlation-based heuristic
        var ordering = DetermineOrdering(data, d);

        // Parent sets for each variable
        var parentSets = new HashSet<int>[d];
        for (int i = 0; i < d; i++)
            parentSets[i] = [];

        // For each variable in the ordering, greedily add parents
        for (int idx = 1; idx < d; idx++)
        {
            int target = ordering[idx];
            var currentParents = new HashSet<int>(parentSets[target]);
            double currentScore = ComputeBIC(data, target, currentParents);

            bool improved = true;
            while (improved && currentParents.Count < _maxParentsPerNode)
            {
                improved = false;
                int bestCandidate = -1;
                double bestScore = currentScore;

                // Consider all predecessors in the ordering as potential parents
                for (int predIdx = 0; predIdx < idx; predIdx++)
                {
                    int candidate = ordering[predIdx];
                    if (currentParents.Contains(candidate)) continue;

                    var testParents = new HashSet<int>(currentParents) { candidate };
                    double score = ComputeBIC(data, target, testParents);

                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestCandidate = candidate;
                    }
                }

                if (bestCandidate >= 0)
                {
                    currentParents.Add(bestCandidate);
                    currentScore = bestScore;
                    improved = true;
                }
            }

            parentSets[target] = currentParents;
        }

        // Build adjacency matrix from parent sets
        var W = new Matrix<T>(d, d);
        for (int j = 0; j < d; j++)
        {
            foreach (int parent in parentSets[j])
            {
                // Estimate edge weight via regression coefficient
                double weight = EstimateEdgeWeight(data, parent, j, parentSets[j]);
                W[parent, j] = NumOps.FromDouble(weight);
            }
        }

        return ThresholdMatrix(W, 0.0); // K2 edges are already selected; keep all
    }

    /// <summary>
    /// Determines a variable ordering using a correlation-based heuristic.
    /// Variables with the least incoming correlations are placed first (likely root causes).
    /// </summary>
    private int[] DetermineOrdering(Matrix<T> data, int d)
    {
        // Compute total absolute correlation for each variable
        var totalCorr = new double[d];
        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                double c = ComputeAbsCorrelation(data, i, j);
                totalCorr[i] += c;
                totalCorr[j] += c;
            }
        }

        // Sort: variables with least total correlation first (likely exogenous/root causes)
        var ordering = Enumerable.Range(0, d).ToArray();
        Array.Sort(ordering, (a, b) => totalCorr[a].CompareTo(totalCorr[b]));
        return ordering;
    }

    /// <summary>
    /// Estimates the edge weight for parent→child using OLS regression coefficient.
    /// </summary>
    private double EstimateEdgeWeight(Matrix<T> data, int parent, int child, HashSet<int> allParents)
    {
        int n = data.Rows;
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
            double dc = NumOps.ToDouble(data[i, child]) - meanC;
            cov += dp * dc;
            varP += dp * dp;
        }

        return varP > 1e-10 ? cov / varP : 0;
    }
}
