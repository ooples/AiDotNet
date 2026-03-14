using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Hybrid;

/// <summary>
/// GFCI — Greedy FCI, a hybrid of GES and FCI.
/// </summary>
/// <remarks>
/// <para>
/// GFCI combines score-based and constraint-based approaches to handle latent confounders.
/// Phase 1 uses greedy score-based search (like GES) to find an initial skeleton and
/// orientations. Phase 2 applies FCI-like rules to detect possible latent confounders
/// and convert directed edges to bidirected edges where appropriate.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item><b>Score-based phase:</b> Use greedy hill climbing with BIC to find an initial DAG</item>
/// <item><b>Skeleton validation:</b> Test remaining edges with CI tests; remove if independent</item>
/// <item><b>Latent detection:</b> For each unshielded triple i — k — j where i and j are
///   non-adjacent, check if k is in the separation set. If not, orient as collider (i → k ← j)</item>
/// <item><b>Discriminating path check:</b> Identify possible bidirected edges (latent confounders)
///   by checking if edges oriented in both directions can be explained by a common cause</item>
/// <item>Apply FCI orientation rules to propagate orientations</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> GFCI is useful when you suspect there are hidden variables affecting
/// your data. It first quickly finds a good graph structure, then checks whether some
/// connections might actually be due to hidden common causes rather than direct effects.
/// </para>
/// <para>
/// Reference: Ogarrio et al. (2016), "A Hybrid Causal Search Algorithm for Latent
/// Variable Models", PGM.
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
[ModelPaper("A Hybrid Causal Search Algorithm for Latent Variable Models", "https://doi.org/10.48550/arXiv.1602.01426", Year = 2016, Authors = "Juan Miguel Ogarrio, Peter Spirtes, Joe Ramsey")]
public class GFCIAlgorithm<T> : HybridBase<T>
{
    private readonly int _maxConditioningSetSize;

    /// <inheritdoc/>
    public override string Name => "GFCI";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => true;

    /// <summary>
    /// Initializes GFCI with optional configuration.
    /// </summary>
    public GFCIAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyHybridOptions(options);
        _maxConditioningSetSize = options?.MaxConditioningSetSize ?? 3;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        // Phase 1: Score-based initial DAG via greedy hill climbing
        var parentSets = new HashSet<int>[d];
        for (int i = 0; i < d; i++) parentSets[i] = [];

        bool improved = true;
        while (improved)
        {
            improved = false;
            int bestTarget = -1, bestParent = -1;
            double bestDelta = 0;

            for (int target = 0; target < d; target++)
            {
                if (parentSets[target].Count >= MaxParents) continue;

                for (int candidate = 0; candidate < d; candidate++)
                {
                    if (candidate == target || parentSets[target].Contains(candidate)) continue;
                    if (WouldCreateCycle(parentSets, candidate, target, d)) continue;

                    var oldScore = ComputeBIC(data, target, parentSets[target].ToList());
                    var newParents = new List<int>(parentSets[target]) { candidate };
                    var newScore = ComputeBIC(data, target, newParents);

                    double delta = oldScore - newScore;
                    if (delta > bestDelta)
                    {
                        bestDelta = delta;
                        bestTarget = target;
                        bestParent = candidate;
                    }
                }
            }

            if (bestTarget >= 0 && bestDelta > 0)
            {
                parentSets[bestTarget].Add(bestParent);
                improved = true;
            }
        }

        // Build skeleton from parent sets
        var adj = new bool[d, d];
        var oriented = new bool[d, d];
        for (int child = 0; child < d; child++)
        {
            foreach (int parent in parentSets[child])
            {
                adj[parent, child] = true;
                adj[child, parent] = true;
                oriented[parent, child] = true; // initially directed by score-based phase
            }
        }

        // Phase 2: Validate skeleton with CI tests
        var sepSets = new Dictionary<(int, int), List<int>>();

        for (int condSize = 0; condSize <= _maxConditioningSetSize; condSize++)
        {
            for (int i = 0; i < d; i++)
            {
                for (int j = i + 1; j < d; j++)
                {
                    if (!adj[i, j]) continue;

                    var neighbors = new List<int>();
                    for (int k = 0; k < d; k++)
                        if (k != i && k != j && adj[i, k]) neighbors.Add(k);

                    if (neighbors.Count < condSize) continue;

                    foreach (var condSet in GetCombinations(neighbors, condSize))
                    {
                        if (IsCI(data, i, j, condSet, n))
                        {
                            adj[i, j] = false;
                            adj[j, i] = false;
                            oriented[i, j] = false;
                            oriented[j, i] = false;
                            sepSets[(i, j)] = condSet;
                            sepSets[(j, i)] = condSet;
                            break;
                        }
                    }
                }
            }
        }

        // Phase 3: FCI-like orientation — detect latent confounders
        var bidirected = new bool[d, d];

        // Re-orient v-structures based on separation sets
        for (int k = 0; k < d; k++)
        {
            for (int i = 0; i < d; i++)
            {
                if (i == k || !adj[i, k]) continue;
                for (int j = i + 1; j < d; j++)
                {
                    if (j == k || !adj[j, k] || adj[i, j]) continue;

                    // Only orient collider if we have a recorded separator for (i,j)
                    // and k is NOT in it. Without a separator, we can't determine collider status.
                    if (!sepSets.TryGetValue((i, j), out var sepSet))
                        continue; // No separator found — skip
                    if (sepSet.Contains(k))
                        continue;

                    oriented[i, k] = true;
                    oriented[j, k] = true;

                    // Check if both i→k and j→k conflict with existing orientations
                    if (oriented[k, i])
                    {
                        bidirected[i, k] = true;
                        bidirected[k, i] = true;
                        oriented[i, k] = false;
                        oriented[k, i] = false;
                    }
                    if (oriented[k, j])
                    {
                        bidirected[j, k] = true;
                        bidirected[k, j] = true;
                        oriented[j, k] = false;
                        oriented[k, j] = false;
                    }
                }
            }
        }

        // Build weighted adjacency matrix
        var W = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (!adj[i, j]) continue;

                if (bidirected[i, j])
                {
                    double weight = Math.Abs(ComputeCorrelation(data, i, j));
                    W[i, j] = NumOps.FromDouble(weight);
                }
                else if (oriented[i, j])
                {
                    double weight = ComputeOLSWeight(data, i, j, n);
                    W[i, j] = NumOps.FromDouble(weight);
                }
                else if (!oriented[j, i] && !bidirected[j, i])
                {
                    W[i, j] = NumOps.FromDouble(Math.Abs(ComputeCorrelation(data, i, j)));
                }
            }
        }

        return W;
    }

    private bool IsCI(Matrix<T> data, int i, int j, List<int> condSet, int n)
    {
        double partialCorr = ComputePartialCorrelation(data, i, j, condSet, n);
        int dof = n - condSet.Count - 3;
        if (dof <= 0) return true;

        double clamped = Math.Max(-0.999999, Math.Min(partialCorr, 0.999999));
        double z = Math.Sqrt(dof) * 0.5 * Math.Log((1 + clamped) / (1 - clamped));
        double pValue = 2 * (1 - NormalCDF(Math.Abs(z)));
        return pValue > Alpha;
    }

    private double ComputePartialCorrelation(Matrix<T> data, int i, int j, List<int> condSet, int n)
    {
        if (condSet.Count == 0) return ComputeCorrelation(data, i, j);

        int p = condSet.Count;

        // Build all residuals simultaneously using multivariate OLS
        // to avoid order-dependent sequential residualization
        double meanI = 0, meanJ = 0;
        for (int k = 0; k < n; k++)
        {
            meanI += NumOps.ToDouble(data[k, i]);
            meanJ += NumOps.ToDouble(data[k, j]);
        }
        meanI /= n; meanJ /= n;

        // Compute conditioning variable means
        var condMeans = new double[p];
        for (int ci = 0; ci < p; ci++)
        {
            for (int k = 0; k < n; k++)
                condMeans[ci] += NumOps.ToDouble(data[k, condSet[ci]]);
            condMeans[ci] /= n;
        }

        // Build normal equations: X'X (p x p) and X'y for both i and j
        var XtX = new double[p, p];
        var XtI = new double[p];
        var XtJ = new double[p];

        for (int k = 0; k < n; k++)
        {
            var dx = new double[p];
            for (int ci = 0; ci < p; ci++)
                dx[ci] = NumOps.ToDouble(data[k, condSet[ci]]) - condMeans[ci];
            double di = NumOps.ToDouble(data[k, i]) - meanI;
            double dj = NumOps.ToDouble(data[k, j]) - meanJ;

            for (int a = 0; a < p; a++)
            {
                XtI[a] += dx[a] * di;
                XtJ[a] += dx[a] * dj;
                for (int b = a; b < p; b++)
                    XtX[a, b] += dx[a] * dx[b];
            }
        }
        for (int a = 0; a < p; a++)
        {
            XtX[a, a] += 1e-10; // Ridge
            for (int b = a + 1; b < p; b++)
                XtX[b, a] = XtX[a, b];
        }

        // Solve for coefficients via Gaussian elimination
        var betaI = SolveSmallSystem(XtX, XtI, p);
        var betaJ = SolveSmallSystem(XtX, XtJ, p);

        // Compute residuals
        var residI = new double[n];
        var residJ = new double[n];
        for (int k = 0; k < n; k++)
        {
            residI[k] = NumOps.ToDouble(data[k, i]) - meanI;
            residJ[k] = NumOps.ToDouble(data[k, j]) - meanJ;
            for (int ci = 0; ci < p; ci++)
            {
                double dc = NumOps.ToDouble(data[k, condSet[ci]]) - condMeans[ci];
                residI[k] -= betaI[ci] * dc;
                residJ[k] -= betaJ[ci] * dc;
            }
        }

        double sxy = 0, sxx = 0, syy = 0;
        for (int k = 0; k < n; k++)
        {
            sxy += residI[k] * residJ[k];
            sxx += residI[k] * residI[k];
            syy += residJ[k] * residJ[k];
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    // SolveSmallSystem is inherited from CausalDiscoveryBase<T>

    private double ComputeOLSWeight(Matrix<T> data, int parent, int child, int n)
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

        return varP > 1e-10 ? cov / varP : 0;
    }

    private static IEnumerable<List<int>> GetCombinations(List<int> items, int size)
    {
        if (size == 0) { yield return []; yield break; }
        for (int i = 0; i <= items.Count - size; i++)
        {
            foreach (var rest in GetCombinations(items.Skip(i + 1).ToList(), size - 1))
            {
                var combination = new List<int> { items[i] };
                combination.AddRange(rest);
                yield return combination;
            }
        }
    }

    private static bool WouldCreateCycle(HashSet<int>[] parents, int from, int to, int d)
    {
        var visited = new HashSet<int>();
        var queue = new Queue<int>();
        queue.Enqueue(from);
        while (queue.Count > 0)
        {
            int current = queue.Dequeue();
            if (current == to) return true;
            if (!visited.Add(current)) continue;
            foreach (int p in parents[current]) queue.Enqueue(p);
        }
        return false;
    }

    private static double NormalCDF(double x)
    {
        double a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
        double a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x) / Math.Sqrt(2);
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
        return 0.5 * (1.0 + sign * y);
    }
}
