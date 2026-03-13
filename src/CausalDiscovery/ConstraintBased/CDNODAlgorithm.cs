using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ConstraintBased;

/// <summary>
/// CD-NOD — Constraint-based Discovery from Non-stationary / heterogeneous Data.
/// </summary>
/// <remarks>
/// <para>
/// CD-NOD extends constraint-based causal discovery to handle data from changing
/// environments or multiple domains. It augments the data with a context variable C
/// (representing time index or domain indicator), then uses conditional independence
/// tests to detect which causal mechanisms change across contexts and leverages these
/// changes to orient more edges than standard PC.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Augment data with a context variable C (row index normalized to [0,1])</item>
/// <item>Run PC skeleton phase on augmented data (d+1 variables)</item>
/// <item>Identify variables connected to C (these have changing distributions)</item>
/// <item>For edges where one endpoint changes and the other doesn't:
///   orient non-changing → changing (cause is stable, effect changes)</item>
/// <item>Apply standard v-structure and Meek rules on remaining edges</item>
/// <item>Return the d x d adjacency matrix (dropping the context variable)</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> CD-NOD exploits the idea that when conditions change, causes
/// tend to stay the same while effects change. By adding a "time" or "context" variable,
/// it can detect which relationships shift, giving extra information for causal direction.
/// </para>
/// <para>
/// Reference: Huang et al. (2020), "Causal Discovery from Heterogeneous/Nonstationary Data", JMLR.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("Causal Discovery from Heterogeneous/Nonstationary Data", "https://jmlr.org/papers/v21/19-232.html", Year = 2020, Authors = "Biwei Huang, Kun Zhang, Jiji Zhang, Joseph Ramsey, Ruben Sanchez-Romero, Clark Glymour, Bernhard Scholkopf")]
public class CDNODAlgorithm<T> : ConstraintBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "CD-NOD";

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => false;

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes CD-NOD with optional configuration.
    /// </summary>
    public CDNODAlgorithm(CausalDiscoveryOptions? options = null) { ApplyConstraintOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        // Augment data with context variable C (normalized row index)
        int dAug = d + 1;
        var augData = new Matrix<T>(n, dAug);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
                augData[i, j] = data[i, j];
            // Context variable = normalized row index
            augData[i, d] = NumOps.FromDouble((double)i / (n - 1));
        }

        int contextIdx = d; // index of context variable

        // Phase 1: Skeleton discovery on augmented data
        var adj = new bool[dAug, dAug];
        for (int i = 0; i < dAug; i++)
            for (int j = 0; j < dAug; j++)
                adj[i, j] = (i != j);

        var sepSets = new Dictionary<(int, int), List<int>>();

        for (int condSize = 0; condSize <= MaxConditioningSetSize; condSize++)
        {
            for (int i = 0; i < dAug; i++)
            {
                for (int j = i + 1; j < dAug; j++)
                {
                    if (!adj[i, j]) continue;

                    var neighbors = GetNeighbors(adj, i, j, dAug);
                    if (neighbors.Count < condSize) continue;

                    foreach (var condSet in GetCombinations(neighbors, condSize))
                    {
                        if (TestCI(augData, i, j, condSet, Alpha))
                        {
                            adj[i, j] = false;
                            adj[j, i] = false;
                            sepSets[(i, j)] = condSet;
                            sepSets[(j, i)] = condSet;
                            break;
                        }
                    }
                }
            }
        }

        // Phase 2: Identify non-stationary variables (connected to C)
        var connectedToContext = new HashSet<int>();
        for (int i = 0; i < d; i++)
        {
            if (adj[i, contextIdx])
                connectedToContext.Add(i);
        }

        // Phase 3: Orient edges using non-stationarity
        var oriented = new bool[dAug, dAug];

        // If i is NOT connected to C (stable) and j IS connected to C (changing),
        // orient i → j (stable cause → changing effect)
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (i == j || !adj[i, j]) continue;
                if (!connectedToContext.Contains(i) && connectedToContext.Contains(j))
                {
                    oriented[i, j] = true;
                }
            }
        }

        // Phase 4: Standard v-structure orientation for remaining
        for (int k = 0; k < d; k++)
        {
            for (int i = 0; i < d; i++)
            {
                if (i == k || !adj[i, k]) continue;
                for (int j = i + 1; j < d; j++)
                {
                    if (j == k || !adj[j, k] || adj[i, j]) continue;

                    if (sepSets.TryGetValue((i, j), out var sepSet) && sepSet.Contains(k))
                        continue;

                    if (!oriented[k, i] && !oriented[k, j])
                    {
                        oriented[i, k] = true;
                        oriented[j, k] = true;
                    }
                }
            }
        }

        // Phase 5: Meek rules
        ApplyMeekRules(adj, oriented, d);

        // Build d x d adjacency matrix (drop context variable)
        var W = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (!adj[i, j]) continue;

                // Include edge if oriented i→j or undirected (skip if oriented j→i)
                if (!oriented[j, i])
                    W[i, j] = NumOps.FromDouble(Math.Abs(ComputePartialCorr(data, i, j, [])));
            }
        }

        return W;
    }

    private static void ApplyMeekRules(bool[,] adj, bool[,] oriented, int d)
    {
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    if (!adj[i, j] || oriented[i, j] || oriented[j, i]) continue;

                    for (int k = 0; k < d; k++)
                    {
                        if (k == i || k == j) continue;
                        if (oriented[k, i] && !adj[k, j])
                        {
                            oriented[i, j] = true;
                            changed = true;
                            break;
                        }
                    }
                    if (oriented[i, j]) continue;

                    // Meek Rule 2: i — j and i → k → j ⇒ orient i → j
                    for (int k = 0; k < d; k++)
                    {
                        if (k == i || k == j) continue;
                        if (oriented[i, k] && oriented[k, j])
                        {
                            oriented[i, j] = true;
                            changed = true;
                            break;
                        }
                    }
                    if (oriented[i, j]) continue;

                    // Meek Rule 3: i — j, i — k1, i — k2, k1 → j, k2 → j, k1 and k2 non-adjacent
                    bool rule3Applied = false;
                    for (int k1 = 0; k1 < d && !rule3Applied; k1++)
                    {
                        if (k1 == i || k1 == j) continue;
                        if (!adj[i, k1] || oriented[i, k1] || oriented[k1, i]) continue; // i — k1 undirected
                        if (!oriented[k1, j]) continue; // k1 → j

                        for (int k2 = k1 + 1; k2 < d && !rule3Applied; k2++)
                        {
                            if (k2 == i || k2 == j) continue;
                            if (!adj[i, k2] || oriented[i, k2] || oriented[k2, i]) continue; // i — k2 undirected
                            if (!oriented[k2, j]) continue; // k2 → j
                            if (adj[k1, k2]) continue; // k1, k2 must be non-adjacent

                            oriented[i, j] = true;
                            changed = true;
                            rule3Applied = true;
                        }
                    }
                    if (oriented[i, j]) continue;

                    // Meek Rule 4: i — j, i — k, k → l → j, j NOT adj k ⇒ orient i → j
                    for (int k = 0; k < d; k++)
                    {
                        if (k == i || k == j) continue;
                        if (!adj[i, k] || oriented[i, k] || oriented[k, i]) continue; // i — k undirected
                        if (adj[j, k]) continue; // j must NOT be adjacent to k

                        bool found = false;
                        for (int l = 0; l < d; l++)
                        {
                            if (l == i || l == j || l == k) continue;
                            if (oriented[k, l] && oriented[l, j])
                            {
                                found = true;
                                break;
                            }
                        }
                        if (found)
                        {
                            oriented[i, j] = true;
                            changed = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    private static List<int> GetNeighbors(bool[,] adj, int i, int j, int d)
    {
        var neighbors = new List<int>();
        for (int k = 0; k < d; k++)
            if (k != i && k != j && (adj[i, k] || adj[j, k]))
                neighbors.Add(k);
        return neighbors;
    }
}
