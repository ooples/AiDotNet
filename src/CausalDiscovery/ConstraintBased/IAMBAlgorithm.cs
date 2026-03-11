using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ConstraintBased;

/// <summary>
/// IAMB (Incremental Association Markov Blanket) — efficient Markov blanket discovery.
/// </summary>
/// <remarks>
/// <para>
/// IAMB discovers the causal structure by first finding the Markov blanket of each variable
/// using a two-phase approach:
/// <list type="number">
/// <item><b>Forward phase:</b> Greedily add the variable most associated with the target
///   (given the current blanket) until no more significantly associated variables exist</item>
/// <item><b>Backward phase:</b> Remove any variable from the blanket that becomes
///   conditionally independent of the target given the remaining blanket members</item>
/// <item>Build the skeleton from pairwise Markov blanket membership</item>
/// <item>Orient edges using v-structure detection and Meek rules</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> IAMB finds the "neighborhood" (Markov blanket) of each variable —
/// the set of variables that directly influence it or are directly influenced by it.
/// It does this by adding the most relevant variables one at a time, then pruning any
/// that turn out to be redundant. The blankets are then combined into a causal graph.
/// </para>
/// <para>
/// Reference: Tsamardinos et al. (2003), "Algorithms for Large Scale Markov Blanket Discovery".
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
[ModelPaper("Algorithms for Large Scale Markov Blanket Discovery", "https://www.aaai.org/Papers/FLAIRS/2003/Flairs03-073.pdf", Year = 2003, Authors = "Ioannis Tsamardinos, Constantin F. Aliferis, Alexander Statnikov")]
public class IAMBAlgorithm<T> : ConstraintBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "IAMB";

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => false;

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes IAMB with optional configuration.
    /// </summary>
    public IAMBAlgorithm(CausalDiscoveryOptions? options = null) { ApplyConstraintOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int d = data.Columns;

        if (data.Rows < 3 || d < 2) return new Matrix<T>(d, d);

        // Step 1: Find Markov blanket for each variable
        var blankets = new HashSet<int>[d];
        for (int target = 0; target < d; target++)
        {
            blankets[target] = FindMarkovBlanket(data, target, d);
        }

        // Step 2: Build skeleton — edge between i and j if i in MB(j) AND j in MB(i)
        var adj = new bool[d, d];
        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                if (blankets[i].Contains(j) && blankets[j].Contains(i))
                {
                    adj[i, j] = true;
                    adj[j, i] = true;
                }
            }
        }

        // Step 3: Orient v-structures
        var oriented = new bool[d, d];
        var sepSets = new Dictionary<(int, int), List<int>>();

        // Find separation sets for non-adjacent pairs
        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                if (adj[i, j]) continue;
                // Try to find separation set from blanket intersection
                var candidates = new List<int>(blankets[i].Intersect(blankets[j]));
                for (int size = 0; size <= Math.Min(MaxConditioningSetSize, candidates.Count); size++)
                {
                    bool found = false;
                    foreach (var condSet in GetCombinations(candidates, size))
                    {
                        if (TestCI(data, i, j, condSet, Alpha))
                        {
                            sepSets[(i, j)] = condSet;
                            sepSets[(j, i)] = condSet;
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }
            }
        }

        // Orient v-structures
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

                    oriented[i, k] = true;
                    oriented[j, k] = true;
                }
            }
        }

        // Step 4: Meek rules
        ApplyMeekRules(adj, oriented, d);

        // Build weighted adjacency matrix
        var W = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (!adj[i, j]) continue;
                if (oriented[i, j])
                    W[i, j] = NumOps.FromDouble(Math.Abs(ComputePartialCorr(data, i, j, [])));
                else if (!oriented[j, i])
                    W[i, j] = NumOps.FromDouble(Math.Abs(ComputePartialCorr(data, i, j, [])));
            }
        }

        return W;
    }

    /// <summary>
    /// Finds the Markov blanket of a target variable using IAMB's forward-backward approach.
    /// </summary>
    private HashSet<int> FindMarkovBlanket(Matrix<T> data, int target, int d)
    {
        var blanket = new HashSet<int>();

        // Forward phase: add most associated variable until none remain significant
        bool added = true;
        while (added)
        {
            added = false;
            int bestCandidate = -1;
            double bestAssociation = 0;

            for (int candidate = 0; candidate < d; candidate++)
            {
                if (candidate == target || blanket.Contains(candidate)) continue;

                // Measure association: |partial correlation| given current blanket
                double assoc = Math.Abs(ComputePartialCorr(data, target, candidate, blanket.ToList()));

                if (assoc > bestAssociation)
                {
                    bestAssociation = assoc;
                    bestCandidate = candidate;
                }
            }

            if (bestCandidate >= 0)
            {
                // Test if significantly associated
                var condList = blanket.ToList();
                bool independent = TestCI(data, target, bestCandidate, condList, Alpha);

                if (!independent)
                {
                    blanket.Add(bestCandidate);
                    added = true;
                }
            }
        }

        // Backward phase: remove any variable that is CI given the rest
        var toRemove = new List<int>();
        foreach (int member in blanket)
        {
            var rest = blanket.Where(v => v != member).ToList();
            if (TestCI(data, target, member, rest, Alpha))
            {
                toRemove.Add(member);
            }
        }

        foreach (int member in toRemove)
            blanket.Remove(member);

        return blanket;
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
