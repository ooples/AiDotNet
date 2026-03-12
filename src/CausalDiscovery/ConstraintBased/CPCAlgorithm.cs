using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ConstraintBased;

/// <summary>
/// CPC (Conservative PC) — PC variant that avoids erroneous v-structure orientation.
/// </summary>
/// <remarks>
/// <para>
/// CPC modifies the PC algorithm's orientation phase to be more conservative. Before
/// orienting a triple i — k — j as a v-structure (i → k ← j), CPC checks ALL possible
/// subsets of the adjacency of i (and j) that could serve as separation sets. A v-structure
/// is only oriented if k is NEVER in any separation set (definite non-collider) or ALWAYS
/// in every separation set (definite collider). Ambiguous triples are left unoriented.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Run PC skeleton phase (same as standard PC)</item>
/// <item>For each unshielded triple i — k — j:</item>
/// <item>  Collect ALL subsets of adj(i)\{j} and adj(j)\{i} up to MaxConditioningSetSize</item>
/// <item>  Test CI(i, j | S) for each subset S</item>
/// <item>  Classify k as: definite collider (never in any separating set),
///   definite non-collider (always in every separating set), or ambiguous</item>
/// <item>  Only orient v-structure if k is a definite collider</item>
/// <item>Apply Meek orientation rules</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Standard PC sometimes incorrectly orients edges when the
/// independence tests are noisy. CPC is more careful — it only orients an edge
/// when ALL evidence agrees on the direction. This means fewer edges are oriented,
/// but the orientations that remain are more reliable.
/// </para>
/// <para>
/// Reference: Ramsey et al. (2012), "Adjacency-Faithfulness and Conservative
/// Causal Inference", UAI.
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
[ModelPaper("Adjacency-Faithfulness and Conservative Causal Inference", "https://arxiv.org/abs/1206.6843", Year = 2012, Authors = "Joseph Ramsey, Jiji Zhang, Peter Spirtes")]
public class CPCAlgorithm<T> : ConstraintBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "Conservative PC";

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => false;

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes CPC with optional configuration.
    /// </summary>
    public CPCAlgorithm(CausalDiscoveryOptions? options = null) { ApplyConstraintOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int d = data.Columns;

        // Phase 1: Skeleton discovery (same as PC)
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

                    var neighbors = GetNeighbors(adj, i, j, d);
                    if (neighbors.Count < condSize) continue;

                    foreach (var condSet in GetCombinations(neighbors, condSize))
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

        // Phase 2: Conservative v-structure orientation
        var oriented = new bool[d, d];

        for (int k = 0; k < d; k++)
        {
            for (int i = 0; i < d; i++)
            {
                if (i == k || !adj[i, k]) continue;
                for (int j = i + 1; j < d; j++)
                {
                    if (j == k || !adj[j, k] || adj[i, j]) continue;

                    // Unshielded triple: i — k — j with i and j non-adjacent
                    // Check ALL separating subsets to classify k
                    bool isDefiniteCollider = IsDefiniteCollider(data, i, j, k, adj, d);

                    if (isDefiniteCollider)
                    {
                        oriented[i, k] = true;
                        oriented[j, k] = true;
                    }
                    // Otherwise leave unoriented (conservative approach)
                }
            }
        }

        // Phase 3: Apply Meek rules
        ApplyMeekRules(adj, oriented, d);

        // Build weighted adjacency matrix
        var W = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (!adj[i, j]) continue;

                if (oriented[i, j])
                {
                    W[i, j] = NumOps.FromDouble(Math.Abs(ComputePartialCorr(data, i, j, [])));
                }
                else if (oriented[j, i])
                {
                    continue; // j→i, skip
                }
                else
                {
                    W[i, j] = NumOps.FromDouble(Math.Abs(ComputePartialCorr(data, i, j, [])));
                }
            }
        }

        return W;
    }

    /// <summary>
    /// Checks if k is a definite collider in the triple i — k — j.
    /// k is a definite collider if no separating set of (i,j) contains k.
    /// </summary>
    private bool IsDefiniteCollider(Matrix<T> data, int i, int j, int k, bool[,] adj, int d)
    {
        bool foundSepSet = false;
        bool kAlwaysInSepSet = true;

        // Check all subsets of adj(i)\{j} and adj(j)\{i}
        var adjI = new List<int>();
        var adjJ = new List<int>();
        for (int v = 0; v < d; v++)
        {
            if (v != j && adj[i, v]) adjI.Add(v);
            if (v != i && adj[j, v]) adjJ.Add(v);
        }

        // Union of potential conditioning variables
        var candidates = new HashSet<int>(adjI);
        foreach (int v in adjJ) candidates.Add(v);

        for (int size = 0; size <= Math.Min(MaxConditioningSetSize, candidates.Count); size++)
        {
            foreach (var condSet in GetCombinations(candidates.ToList(), size))
            {
                if (TestCI(data, i, j, condSet, Alpha))
                {
                    foundSepSet = true;
                    if (!condSet.Contains(k))
                    {
                        kAlwaysInSepSet = false;
                    }
                }
            }
        }

        // k is definite collider if we found separating sets and k was never in any of them
        return foundSepSet && !kAlwaysInSepSet;
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

                    // R1: k→i — j, k and j non-adjacent → i→j
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

                    // R2: i→k→j → i→j
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

    private static List<int> GetNeighbors(bool[,] adj, int i, int j, int d)
    {
        var neighbors = new List<int>();
        for (int k = 0; k < d; k++)
            if (k != i && k != j && (adj[i, k] || adj[j, k]))
                neighbors.Add(k);
        return neighbors;
    }
}
