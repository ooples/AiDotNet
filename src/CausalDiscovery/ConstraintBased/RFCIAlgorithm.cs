using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ConstraintBased;

/// <summary>
/// RFCI (Really Fast Causal Inference) — scalable FCI for large datasets.
/// </summary>
/// <remarks>
/// <para>
/// RFCI speeds up FCI by reducing the number of conditional independence tests.
/// Instead of testing all possible conditioning sets (as FCI does), RFCI:
/// <list type="number">
/// <item>Runs a PC-like skeleton phase with limited conditioning set sizes</item>
/// <item>For possible v-structures, only tests conditioning sets from adjacencies
///   (not all possible subsets), reducing complexity from exponential to polynomial</item>
/// <item>Uses discriminating path rules more sparingly</item>
/// <item>Produces a PAG (Partial Ancestral Graph) that accounts for latent confounders</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> RFCI does the same thing as FCI (finds causal relationships
/// even with hidden variables) but much faster. It achieves this by being smarter about
/// which statistical tests to run, skipping tests that are unlikely to change the result.
/// </para>
/// <para>
/// Reference: Colombo et al. (2012), "Learning High-Dimensional DAGs with Latent
/// and Selection Variables", AOAS.
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
[ModelPaper("Learning High-Dimensional Directed Acyclic Graphs with Latent and Selection Variables", "https://doi.org/10.1214/11-AOS940", Year = 2012, Authors = "Diego Colombo, Marloes H. Maathuis, Markus Kalisch, Thomas S. Richardson")]
public class RFCIAlgorithm<T> : ConstraintBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "RFCI";

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => true;

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes RFCI with optional configuration.
    /// </summary>
    public RFCIAlgorithm(CausalDiscoveryOptions? options = null) { ApplyConstraintOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int d = data.Columns;

        if (d < 2)
            throw new ArgumentException($"RFCI requires at least 2 variables, got {d}.");
        if (data.Rows < 3)
            throw new ArgumentException($"RFCI requires at least 3 samples, got {data.Rows}.");

        // Phase 1: Skeleton discovery (PC-style but with limited conditioning)
        var adj = new bool[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                adj[i, j] = (i != j);

        var sepSets = new Dictionary<(int, int), List<int>>();

        for (int condSize = 0; condSize <= MaxConditioningSetSize; condSize++)
        {
            for (int i = 0; i < d; i++)
            {
                for (int j = i + 1; j < d; j++)
                {
                    if (!adj[i, j]) continue;

                    // RFCI key difference: only use adjacencies of i (not union with j)
                    var neighborsI = GetAdjacencies(adj, i, j, d);
                    if (neighborsI.Count >= condSize)
                    {
                        foreach (var condSet in GetCombinations(neighborsI, condSize))
                        {
                            if (TestCI(data, i, j, condSet, Alpha))
                            {
                                adj[i, j] = false;
                                adj[j, i] = false;
                                sepSets[(i, j)] = condSet;
                                sepSets[(j, i)] = condSet;
                                break;
                            }
                        }
                    }

                    if (adj[i, j]) // If not yet removed, also try adjacencies of j
                    {
                        var neighborsJ = GetAdjacencies(adj, j, i, d);
                        if (neighborsJ.Count >= condSize)
                        {
                            foreach (var condSet in GetCombinations(neighborsJ, condSize))
                            {
                                if (TestCI(data, i, j, condSet, Alpha))
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
            }
        }

        // Phase 2: Orient v-structures (colliders)
        // For RFCI: only check adjacency-based conditioning sets
        var oriented = new bool[d, d];
        var bidirected = new bool[d, d]; // marks latent confounder edges

        for (int k = 0; k < d; k++)
        {
            for (int i = 0; i < d; i++)
            {
                if (i == k || !adj[i, k]) continue;
                for (int j = i + 1; j < d; j++)
                {
                    if (j == k || !adj[j, k] || adj[i, j]) continue;

                    // Unshielded triple i — k — j
                    if (sepSets.TryGetValue((i, j), out var sepSet) && sepSet.Contains(k))
                        continue;

                    // RFCI check: verify the v-structure with a targeted CI test
                    // Test if i ⊥ j | {k} ∪ sepSet
                    var augmentedSet = new List<int>(sepSet ?? []);
                    if (!augmentedSet.Contains(k)) augmentedSet.Add(k);

                    if (!TestCI(data, i, j, augmentedSet, Alpha))
                    {
                        // Not independent even with k → definite collider
                        oriented[i, k] = true;
                        oriented[j, k] = true;
                    }
                }
            }
        }

        // Phase 3: Detect possible latent confounders.
        // 3a: Contradictory orientations indicate latent confounding.
        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                if (!adj[i, j]) continue;

                if (oriented[i, j] && oriented[j, i])
                {
                    bidirected[i, j] = true;
                    bidirected[j, i] = true;
                    oriented[i, j] = false;
                    oriented[j, i] = false;
                }
            }
        }

        // 3b: Discriminating path rule (bounded length for RFCI efficiency).
        // A discriminating path for (b, c) is <a, ..., v, b, c> where:
        //   - a is not adjacent to c
        //   - every intermediate node between a and b is a collider and a parent of c
        //   - b is adjacent to c
        // If b is in sepSet(a, c): b is a non-collider → orient b → c
        // If b is NOT in sepSet(a, c): b is a collider → orient b ← c (and mark bidirected if conflict)
        for (int b = 0; b < d; b++)
        {
            for (int c = 0; c < d; c++)
            {
                if (b == c || !adj[b, c]) continue;
                if (bidirected[b, c]) continue;

                // Search for discriminating paths ending at (b, c)
                // A discriminating path <a, ..., v, b, c> requires:
                //   - a is not adjacent to c
                //   - every intermediate node between a and b is a collider (both neighbors oriented toward it)
                //     and is a parent of c
                //   - b is adjacent to c
                // Use BFS from b backward to find valid discriminating paths
                bool foundPath = false;
                foreach (int v in GetAdjacencies(adj, b, c, d))
                {
                    if (!oriented[v, c]) continue;     // v must be parent of c (v → c)
                    if (!adj[b, v]) continue;           // b adjacent to v
                    if (!oriented[v, b] && !oriented[b, v] && !bidirected[b, v]) continue; // must be oriented or bidirected edge
                    // v must be a collider: needs an incoming edge from the previous node AND oriented[v,b]
                    // For a 2-node path (path = <a, v, b, c>), v is a collider if a→v and b→v (or bidirected)

                    foreach (int a in GetAdjacencies(adj, v, b, d))
                    {
                        if (a == c || a == b || adj[a, c]) continue;
                        if (!oriented[a, v]) continue; // a → v must be oriented

                        // Check v is a collider: a → v ← b (or at least a→v and v adjacent to b)
                        // v is a collider if oriented[a,v] AND (oriented[b,v] or bidirected[b,v])
                        if (!oriented[b, v] && !bidirected[b, v]) continue;

                        if (sepSets.TryGetValue((a, c), out var sepAC))
                        {
                            if (sepAC.Contains(b))
                            {
                                // b is non-collider → orient b → c
                                if (oriented[c, b])
                                {
                                    // Conflict with existing c→b → bidirected
                                    bidirected[b, c] = true;
                                    bidirected[c, b] = true;
                                    oriented[b, c] = false;
                                    oriented[c, b] = false;
                                }
                                else if (!oriented[b, c])
                                {
                                    oriented[b, c] = true;
                                }
                            }
                            else
                            {
                                // b is collider → orient c → b
                                if (oriented[b, c])
                                {
                                    // Conflict with existing b→c → bidirected
                                    bidirected[b, c] = true;
                                    bidirected[c, b] = true;
                                    oriented[b, c] = false;
                                    oriented[c, b] = false;
                                }
                                else if (!oriented[c, b])
                                {
                                    oriented[c, b] = true;
                                }
                            }
                            foundPath = true;
                            break;
                        }
                    }
                    if (foundPath) break;
                }
            }
        }

        // Phase 4: Apply Meek-like orientation rules
        ApplyMeekRules(adj, oriented, d);

        // Build weighted adjacency matrix
        var W = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (!adj[i, j]) continue;

                if (bidirected[i, j])
                {
                    // Latent confounder: represent as bidirected (both directions)
                    double weight = Math.Abs(ComputePartialCorr(data, i, j, []));
                    W[i, j] = NumOps.FromDouble(weight);
                }
                else if (oriented[i, j])
                {
                    W[i, j] = NumOps.FromDouble(Math.Abs(ComputePartialCorr(data, i, j, [])));
                }
                else if (!oriented[j, i])
                {
                    W[i, j] = NumOps.FromDouble(Math.Abs(ComputePartialCorr(data, i, j, [])));
                }
            }
        }

        return W;
    }

    /// <summary>
    /// Gets adjacencies of node i, excluding node exclude.
    /// </summary>
    private static List<int> GetAdjacencies(bool[,] adj, int node, int exclude, int d)
    {
        var result = new List<int>();
        for (int k = 0; k < d; k++)
        {
            if (k != node && k != exclude && adj[node, k])
                result.Add(k);
        }
        return result;
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
                }
            }
        }
    }
}
