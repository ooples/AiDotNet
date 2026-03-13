using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ConstraintBased;

/// <summary>
/// Fast-IAMB — faster variant of IAMB using speculative forward selection.
/// </summary>
/// <remarks>
/// <para>
/// Fast-IAMB accelerates IAMB by adding multiple variables at once in the forward phase
/// (speculative addition), then relying on the backward phase to remove false positives.
/// In each forward step, ALL variables with significant association are added simultaneously
/// rather than just the single best one.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item><b>Speculative forward phase:</b> In each round, test ALL remaining variables
///   for association with the target given the current blanket. Add ALL that are
///   significantly associated (not just the single best)</item>
/// <item><b>Backward phase:</b> For each member of the blanket, test if it becomes
///   conditionally independent of the target given the remaining blanket members.
///   Remove any that become independent</item>
/// <item>Build skeleton and orient edges (same as IAMB)</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Fast-IAMB is like IAMB but more aggressive — instead of
/// adding one variable at a time, it adds ALL relevant variables at once. This is
/// much faster for high-dimensional data but might add some false positives. The
/// backward phase then cleans up by removing variables that were added incorrectly.
/// </para>
/// <para>
/// Reference: Yaramakala and Margaritis (2005), "Speculative Markov Blanket Discovery
/// for Optimal Feature Selection".
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
[ModelPaper("Speculative Markov Blanket Discovery for Optimal Feature Selection", "https://doi.org/10.1109/ICDM.2005.138", Year = 2005, Authors = "Sandeep Yaramakala, Dimitris Margaritis")]
public class FastIAMBAlgorithm<T> : ConstraintBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "Fast-IAMB";

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => false;

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes Fast-IAMB with optional configuration.
    /// </summary>
    public FastIAMBAlgorithm(CausalDiscoveryOptions? options = null) { ApplyConstraintOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int d = data.Columns;

        if (data.Rows < 3 || d < 2) return new Matrix<T>(d, d);

        // Step 1: Find Markov blanket for each variable using speculative approach
        var blankets = new HashSet<int>[d];
        for (int target = 0; target < d; target++)
        {
            blankets[target] = FindMarkovBlanketFast(data, target, d);
        }

        // Step 2: Build skeleton — symmetric blanket membership
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

        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                if (adj[i, j]) continue;
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

        for (int k = 0; k < d; k++)
        {
            for (int i = 0; i < d; i++)
            {
                if (i == k || !adj[i, k]) continue;
                for (int j = i + 1; j < d; j++)
                {
                    if (j == k || !adj[j, k] || adj[i, j]) continue;
                    // Only orient a collider if we have a recorded separator for (i,j)
                    // and k is NOT in that separator. Without a separator, we can't
                    // determine collider status.
                    if (!sepSets.TryGetValue((i, j), out var sepSet))
                        continue; // No separator found — skip, don't orient
                    if (sepSet.Contains(k))
                        continue; // k is in separator — not a collider
                    oriented[i, k] = true;
                    oriented[j, k] = true;
                }
            }
        }

        // Step 4: Meek rules
        ApplyMeekRules(adj, oriented, d);

        // Build adjacency matrix
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
    /// Finds the Markov blanket using speculative (batch) forward selection.
    /// </summary>
    private HashSet<int> FindMarkovBlanketFast(Matrix<T> data, int target, int d)
    {
        var blanket = new HashSet<int>();

        // Speculative forward phase: add ALL significantly associated variables at once
        bool added = true;
        while (added)
        {
            added = false;
            var toAdd = new List<int>();
            var condList = blanket.ToList();

            for (int candidate = 0; candidate < d; candidate++)
            {
                if (candidate == target || blanket.Contains(candidate)) continue;

                // Test if candidate is significantly associated with target given current blanket
                bool independent = TestCI(data, target, candidate, condList, Alpha);
                if (!independent)
                {
                    toAdd.Add(candidate);
                }
            }

            if (toAdd.Count > 0)
            {
                foreach (int v in toAdd)
                    blanket.Add(v);
                added = true;
            }
        }

        // Backward phase: remove false positives
        bool removed = true;
        while (removed)
        {
            removed = false;
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
            {
                blanket.Remove(member);
                removed = true;
            }
        }

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
