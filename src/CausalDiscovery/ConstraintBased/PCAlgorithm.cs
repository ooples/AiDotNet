using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ConstraintBased;

/// <summary>
/// PC Algorithm — constraint-based causal discovery using conditional independence tests.
/// </summary>
/// <remarks>
/// <para>
/// The PC (Peter-Clark) algorithm learns a causal graph by:
/// <list type="number">
/// <item>Starting with a complete undirected graph</item>
/// <item>Removing edges between conditionally independent variable pairs</item>
/// <item>Orienting edges using v-structures and orientation rules</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> PC figures out which variables cause which by testing:
/// "Are X and Y still related after we account for other variables?" If X and Y
/// become unrelated (conditionally independent) given some set of variables, the
/// edge between them is removed. Then remaining edges are oriented to form a DAG.
/// </para>
/// <para>
/// Reference: Spirtes, Glymour, and Scheines (2000), "Causation, Prediction, and Search".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class PCAlgorithm<T> : ConstraintBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "PC Algorithm";

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => false;

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes the PC algorithm with optional configuration.
    /// </summary>
    public PCAlgorithm(CausalDiscoveryOptions? options = null)
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

        // Step 1: Initialize complete undirected graph
        var adj = new bool[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                adj[i, j] = (i != j);

        // Store separation sets for orientation
        var sepSets = new Dictionary<(int, int), List<int>>();

        // Step 2: Edge removal via CI tests
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

        // Step 3: Orient v-structures (i → k ← j where i and j are non-adjacent
        // and k is not in sepSet(i,j))
        var oriented = new bool[d, d]; // directed: oriented[i,j] = true means i→j
        for (int k = 0; k < d; k++)
        {
            for (int i = 0; i < d; i++)
            {
                if (i == k || !adj[i, k]) continue;
                for (int j = i + 1; j < d; j++)
                {
                    if (j == k || !adj[j, k] || adj[i, j]) continue;

                    // Check if k is in sepSet(i,j)
                    if (sepSets.TryGetValue((i, j), out var sepSet) && sepSet.Contains(k))
                        continue;

                    // Orient: i → k ← j
                    oriented[i, k] = true;
                    oriented[j, k] = true;
                }
            }
        }

        // Step 4: Build weighted adjacency matrix
        // Use partial correlation magnitudes as edge weights
        var W = new double[d, d];
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (!adj[i, j]) continue;

                if (oriented[i, j])
                {
                    W[i, j] = Math.Abs(ComputePartialCorr(X, n, i, j, []));
                }
                else if (oriented[j, i])
                {
                    // Edge is j → i, skip i → j direction
                    continue;
                }
                else
                {
                    // Unoriented: assign weight to both directions
                    double weight = Math.Abs(ComputePartialCorr(X, n, i, j, []));
                    W[i, j] = weight;
                }
            }
        }

        return DoubleArrayToMatrix(W);
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
