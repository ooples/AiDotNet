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

        var engine = AiDotNetEngine.Current;
        var relevance = trueRelevance.AsTensor();
        var gain = useExponentialGain
            ? engine.TensorAddScalar(engine.TensorExp2(relevance), NumOps.FromDouble(-1.0))
            : relevance;

        // Discount 1 / log2(rank + 1) for ranks 1..cutoff and zero beyond the cutoff, plus its
        // cumulative sum with a leading zero so any rank range sums by two lookups.
        // Linspace needs two points, so a cutoff of 1 builds its single rank position (2) directly.
        var rankPositions = cutoff == 1
            ? Tensor<T>.CreateDefault(new[] { 1 }, NumOps.FromDouble(2.0))
            : engine.TensorLinspace(NumOps.FromDouble(2.0), NumOps.FromDouble(cutoff + 1.0), cutoff);
        var discounts = engine.TensorReciprocal(engine.TensorLog2(rankPositions));
        if (cutoff < n)
        {
            discounts = engine.TensorConcatenate(new[] { discounts, Tensor<T>.CreateDefault(new[] { n - cutoff }, NumOps.Zero) }, 0);
        }

        var cumulativeDiscounts = engine.TensorConcatenate(
            new[] { Tensor<T>.CreateDefault(new[] { 1 }, NumOps.Zero), engine.TensorCumSum(discounts, 0) }, 0);

        // Ideal DCG: gains in descending order (tied relevances have equal gains, so order within ties is irrelevant).
        var idealGains = engine.TensorSort(gain, -1, descending: true).Values;
        T idcg = engine.TensorSum(engine.TensorMultiply(idealGains, discounts));
        if (NumOps.Equals(idcg, NumOps.Zero)) return NumOps.Zero;

        // DCG with tied predicted scores averaged, as scikit-learn's ndcg_score (ignore_ties=False):
        // every tie group contributes mean(gain over the group) * sum(discounts over the ranks it spans).
        var (_, inverse, counts) = engine.TensorUniqueWithInfo(predictedScores.AsTensor(), sorted: true, returnInverse: true, returnCounts: true);
        if (inverse is null || counts is null)
            throw new InvalidOperationException("Unique with inverse and counts returned no inverse or counts.");

        // Unique values are ascending; ranks run in descending score order.
        var countsDescending = engine.TensorFlip(counts, new[] { 0 });
        var groupEnds = engine.TensorCumSum(countsDescending, 0);
        var groupStarts = engine.TensorSubtract(groupEnds, countsDescending);
        var groupDiscounts = engine.TensorFlip(
            engine.TensorSubtract(engine.TensorGather(cumulativeDiscounts, groupEnds, 0), engine.TensorGather(cumulativeDiscounts, groupStarts, 0)),
            new[] { 0 });

        var groupMeanGains = engine.TensorDivide(engine.ScatterAdd(gain, inverse, 0, counts.Length), counts.Cast<T>());
        T dcg = engine.TensorSum(engine.TensorMultiply(groupMeanGains, groupDiscounts));

        return NumOps.Divide(dcg, idcg);
    }
}
