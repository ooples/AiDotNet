using AiDotNet.Helpers;

namespace AiDotNet.Metrics;

/// <summary>
/// Static helpers for evaluating the quality of a ranking, most notably Normalized Discounted
/// Cumulative Gain (NDCG@k).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> When a model ranks items (stocks, search results, recommendations), you
/// need a number that says "how good is this ordering?". NDCG is the most common such number.
/// It rewards putting the truly-best items near the top and discounts items further down the list
/// (a great item ranked #1 counts for more than the same item ranked #20). It is normalized so
/// that a perfect ranking scores 1.0 and a poor ranking scores closer to 0, which makes scores
/// comparable across groups of different sizes. The "@k" version only looks at the top k positions,
/// which is what you want when you will only act on the top of the list (e.g. only buy the top 10).
/// </para>
/// <para>
/// DCG@k = &#931;_{r=1..k} (2^{gain_r} - 1) / log2(r + 1), where gain_r is the true relevance of the
/// item the model placed at rank r. NDCG@k = DCG@k / IDCG@k, where IDCG@k is the DCG of the ideal
/// ordering (sorting items by true relevance descending).
/// </para>
/// </remarks>
public static class RankingMetrics<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Computes Normalized Discounted Cumulative Gain at cutoff k (NDCG@k) for a single ranking group.
    /// </summary>
    /// <param name="predictedScores">The model's predicted scores; higher means ranked earlier.</param>
    /// <param name="trueRelevance">The true relevance/return of each item (same order as predictions).</param>
    /// <param name="k">
    /// The cutoff: only the top k items (by predicted score) contribute. Use a value &#8804; 0 or
    /// &#8805; the item count to evaluate the full list.
    /// </param>
    /// <param name="useExponentialGain">
    /// When true (default) the gain is 2^relevance - 1 (the standard NDCG used in IR, which sharply
    /// rewards high-relevance items). When false the gain is the raw relevance value, which is
    /// appropriate for signed continuous targets such as forward returns.
    /// </param>
    /// <returns>
    /// NDCG@k in [0, 1] for a perfect ranking of non-negative gains; 1.0 for the ideal ordering and
    /// less for any ordering that misplaces high-relevance items. Returns 0 when the ideal DCG is 0.
    /// </returns>
    /// <exception cref="ArgumentException">Thrown when the input vectors have different lengths.</exception>
    /// <remarks>
    /// For signed targets (e.g. forward returns that can be negative) prefer
    /// <paramref name="useExponentialGain"/> = false, since 2^x for negative x understates large
    /// losers. NDCG with arbitrary signed gains is not bounded to [0, 1] but is still a valid,
    /// monotone "closeness to ideal order" score where 1.0 means the predicted order equals the
    /// ideal order.
    /// </remarks>
    public static T NdcgAtK(Vector<T> predictedScores, Vector<T> trueRelevance, int k, bool useExponentialGain = true)
    {
        if (predictedScores.Length != trueRelevance.Length)
        {
            throw new ArgumentException("Predicted scores and true relevance vectors must have the same length.");
        }

        int n = predictedScores.Length;
        if (n == 0) return NumOps.Zero;

        int cutoff = (k <= 0 || k > n) ? n : k;

        var pred = new double[n];
        var rel = new double[n];
        for (int i = 0; i < n; i++)
        {
            pred[i] = NumOps.ToDouble(predictedScores[i]);
            rel[i] = NumOps.ToDouble(trueRelevance[i]);
        }

        // Order item indices by predicted score (descending). Stable on ties via index tiebreak.
        var byPredicted = StableArgsortDescending(pred);
        // Ideal order: by true relevance (descending).
        var byIdeal = StableArgsortDescending(rel);

        double dcg = Dcg(byPredicted, rel, cutoff, useExponentialGain);
        double idcg = Dcg(byIdeal, rel, cutoff, useExponentialGain);

        if (idcg == 0.0) return NumOps.Zero;
        return NumOps.FromDouble(dcg / idcg);
    }

    /// <summary>
    /// Discounted Cumulative Gain over the first <paramref name="cutoff"/> positions of an ordering.
    /// </summary>
    private static double Dcg(int[] order, double[] relevance, int cutoff, bool useExponentialGain)
    {
        double dcg = 0.0;
        int limit = Math.Min(cutoff, order.Length);
        for (int r = 0; r < limit; r++)
        {
            double gain = useExponentialGain
                ? Math.Pow(2.0, relevance[order[r]]) - 1.0
                : relevance[order[r]];

            // Position r is rank r+1; discount is log2(rank + 1) = log2(r + 2).
            double discount = Math.Log(r + 2.0) / Math.Log(2.0);
            dcg += gain / discount;
        }

        return dcg;
    }

    /// <summary>
    /// Returns indices that sort <paramref name="values"/> in descending order, breaking ties by
    /// original index (so the result is deterministic).
    /// </summary>
    private static int[] StableArgsortDescending(double[] values)
    {
        int n = values.Length;
        var idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;

        Array.Sort(idx, (a, b) =>
        {
            int cmp = values[b].CompareTo(values[a]); // descending
            return cmp != 0 ? cmp : a.CompareTo(b);    // stable tiebreak
        });

        return idx;
    }
}
