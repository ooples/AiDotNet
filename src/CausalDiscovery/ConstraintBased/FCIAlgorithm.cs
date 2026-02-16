using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ConstraintBased;

/// <summary>
/// FCI (Fast Causal Inference) — constraint-based discovery with latent confounders.
/// </summary>
/// <remarks>
/// <para>
/// FCI extends the PC algorithm to handle latent (unmeasured) variables. It outputs a
/// Partial Ancestral Graph (PAG) which uses different edge marks (→, ←→, o→) to indicate
/// whether edges are definitely directed, bidirected (due to a latent common cause), or uncertain.
/// </para>
/// <para>
/// <b>Key Differences from PC:</b>
/// <list type="bullet">
/// <item>Handles latent confounders (hidden common causes)</item>
/// <item>Outputs a PAG, not a CPDAG</item>
/// <item>Uses discriminating path orientation rules</item>
/// <item>Circle marks (o) indicate uncertainty about edge orientation</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Sometimes two variables appear related not because one causes the other,
/// but because a hidden third variable causes both. FCI can detect this pattern. For example,
/// if both ice cream sales and crime rates go up in summer, FCI can infer that the relationship
/// might be due to a hidden variable (temperature) rather than a direct causal link.
/// </para>
/// <para>
/// Reference: Spirtes, Glymour, and Scheines (2000), "Causation, Prediction, and Search".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FCIAlgorithm<T> : ConstraintBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "FCI";

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => true;

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes FCI with optional configuration.
    /// </summary>
    public FCIAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyConstraintOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        var X = new double[n, d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

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

                    var neighbors = GetAllNeighbors(adj, i, d).Where(k => k != j).ToList();
                    if (neighbors.Count < condSize) continue;

                    foreach (var condSet in GetCombinations(neighbors, condSize))
                    {
                        if (TestCI(X, n, i, j, condSet, Alpha))
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

        // Phase 2: Orient v-structures
        // PAG edge types: 0=no edge, 1=tail (—), 2=arrowhead (→), 3=circle (o)
        var edgeMark = new int[d, d]; // edgeMark[i,j] = mark at j end of edge i—j
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                if (adj[i, j]) edgeMark[i, j] = 3; // circle by default

        // Orient unshielded colliders: i *→ k ←* j
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

                    edgeMark[i, k] = 2; // arrowhead at k on i-k edge
                    edgeMark[k, i] = 1; // tail at i on i-k edge
                    edgeMark[j, k] = 2; // arrowhead at k on j-k edge
                    edgeMark[k, j] = 1; // tail at j on j-k edge
                }
            }
        }

        // Phase 3: FCI orientation rules (R1-R4)
        bool changed = true;
        while (changed)
        {
            changed = false;

            // R1: If i *→ j o—* k, and i and k not adjacent, orient j *→ k
            for (int j = 0; j < d; j++)
            {
                for (int i = 0; i < d; i++)
                {
                    if (!adj[i, j] || edgeMark[i, j] != 2) continue;
                    for (int k = 0; k < d; k++)
                    {
                        if (k == i || k == j || !adj[j, k]) continue;
                        if (adj[i, k]) continue;
                        if (edgeMark[j, k] == 3) // circle mark
                        {
                            edgeMark[j, k] = 2;
                            changed = true;
                        }
                    }
                }
            }

            // R2: If i → j *→ k or i *→ j → k, and i *—o k, orient i *→ k
            for (int i = 0; i < d; i++)
            {
                for (int k = 0; k < d; k++)
                {
                    if (!adj[i, k] || edgeMark[i, k] != 3) continue;
                    for (int j = 0; j < d; j++)
                    {
                        if (j == i || j == k || !adj[i, j] || !adj[j, k]) continue;
                        bool case1 = edgeMark[i, j] == 2 && edgeMark[j, i] == 1 && edgeMark[j, k] == 2;
                        bool case2 = edgeMark[i, j] == 2 && edgeMark[j, k] == 2 && edgeMark[k, j] == 1;
                        if (case1 || case2)
                        {
                            edgeMark[i, k] = 2; // arrowhead at k
                            edgeMark[k, i] = 1; // tail at i
                            changed = true;
                        }
                    }
                }
            }

            // R3: If i *→ j ←* k, i *—o l o—* k, i and k not adjacent, l *—o j, orient l *→ j
            for (int j = 0; j < d; j++)
            {
                for (int i = 0; i < d; i++)
                {
                    if (!adj[i, j] || edgeMark[i, j] != 2) continue;
                    for (int k = 0; k < d; k++)
                    {
                        if (k <= i || !adj[k, j] || edgeMark[k, j] != 2 || adj[i, k]) continue;
                        for (int l = 0; l < d; l++)
                        {
                            if (l == i || l == j || l == k) continue;
                            if (!adj[i, l] || !adj[k, l] || !adj[l, j]) continue;
                            if (edgeMark[l, j] != 3) continue;
                            // l is adjacent to both i and k (with circle marks) and l o—o j
                            edgeMark[l, j] = 2; // arrowhead at j
                            edgeMark[j, l] = 1; // tail at l
                            changed = true;
                        }
                    }
                }
            }

            // R4 (simplified discriminating path approximation):
            // The full R4 rule requires finding discriminating paths of arbitrary length,
            // which is O(d!) in the worst case. This simplified version checks the immediate
            // i → j *→ k pattern with i *→ k, covering common short discriminating paths.
            // Full R4 with arbitrary-length path traversal is not yet implemented.
            for (int j = 0; j < d; j++)
            {
                for (int k = 0; k < d; k++)
                {
                    if (!adj[j, k] || edgeMark[j, k] != 3) continue;
                    for (int i = 0; i < d; i++)
                    {
                        if (i == j || i == k) continue;
                        if (!adj[i, j] || edgeMark[i, j] != 2 || edgeMark[j, i] != 1) continue;
                        if (!adj[i, k]) continue;
                        if (edgeMark[j, k] == 3)
                        {
                            edgeMark[j, k] = 2; // arrowhead at k
                            edgeMark[k, j] = 1; // tail at j
                            changed = true;
                        }
                    }
                }
            }
        }

        // Build weighted adjacency from PAG
        var W = new double[d, d];
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (!adj[i, j]) continue;
                if (edgeMark[i, j] == 2) // i → j (arrowhead at j)
                {
                    double weight = Math.Abs(ComputePartialCorr(X, n, i, j, []));
                    W[i, j] = weight;
                }
                else if (edgeMark[i, j] == 3) // uncertain
                {
                    double weight = 0.5 * Math.Abs(ComputePartialCorr(X, n, i, j, []));
                    W[i, j] = weight;
                }
            }
        }

        return DoubleArrayToMatrix(W);
    }

    private static List<int> GetAllNeighbors(bool[,] adj, int node, int d)
    {
        var neighbors = new List<int>();
        for (int k = 0; k < d; k++)
            if (k != node && adj[node, k])
                neighbors.Add(k);
        return neighbors;
    }

}
