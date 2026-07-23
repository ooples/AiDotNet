using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.Evaluation;

/// <summary>
/// Pure-math information-retrieval (IR) metrics over ranked retrieval results: nDCG@k, MRR, MAP,
/// Hit-Rate@k, Precision@k and Recall@k. These compare a ranked list of retrieved document ids
/// against a set of ids known to be relevant. They involve no language model or embedding model and
/// are therefore fully deterministic.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> after your retriever returns a ranked list of documents, these
/// functions grade the ranking. Precision@k asks "of the top k, how many were relevant?"; Recall@k
/// asks "of all relevant documents, how many did we find in the top k?"; MRR rewards putting the
/// first relevant document early; MAP and nDCG reward putting <i>all</i> relevant documents high up,
/// with nDCG additionally discounting lower ranks smoothly.</para>
/// <para>
/// Conventions: Precision@k divides by k (the requested cutoff). Recall@k, MAP, MRR and nDCG return
/// 0 when there are no relevant documents (the quantity is otherwise undefined). nDCG uses binary
/// relevance in the primary overloads and log2 rank discounting.
/// </para>
/// </remarks>
public static class RetrievalMetrics
{
    /// <summary>
    /// Precision at rank k: the fraction of the top-k retrieved items that are relevant.
    /// </summary>
    /// <typeparam name="TId">The document id type.</typeparam>
    /// <param name="ranked">The ranked list of retrieved document ids (best first).</param>
    /// <param name="relevant">The set of relevant document ids.</param>
    /// <param name="k">The rank cutoff (must be positive).</param>
    /// <returns>Precision@k in [0, 1]. Returns 0 when <paramref name="k"/> is not positive.</returns>
    public static double PrecisionAtK<TId>(IReadOnlyList<TId> ranked, ISet<TId> relevant, int k)
    {
        if (ranked == null || relevant == null || k <= 0)
            return 0.0;

        int limit = Math.Min(k, ranked.Count);
        int hits = 0;
        for (int i = 0; i < limit; i++)
        {
            if (relevant.Contains(ranked[i]))
                hits++;
        }

        return (double)hits / k;
    }

    /// <summary>
    /// Recall at rank k: the fraction of all relevant items that appear in the top-k.
    /// </summary>
    /// <typeparam name="TId">The document id type.</typeparam>
    /// <param name="ranked">The ranked list of retrieved document ids (best first).</param>
    /// <param name="relevant">The set of relevant document ids.</param>
    /// <param name="k">The rank cutoff (must be positive).</param>
    /// <returns>Recall@k in [0, 1]. Returns 0 when there are no relevant items or k is not positive.</returns>
    public static double RecallAtK<TId>(IReadOnlyList<TId> ranked, ISet<TId> relevant, int k)
    {
        if (ranked == null || relevant == null || relevant.Count == 0 || k <= 0)
            return 0.0;

        int limit = Math.Min(k, ranked.Count);
        int hits = 0;
        for (int i = 0; i < limit; i++)
        {
            if (relevant.Contains(ranked[i]))
                hits++;
        }

        return (double)hits / relevant.Count;
    }

    /// <summary>
    /// Hit-rate at rank k: 1 if at least one relevant item appears in the top-k, otherwise 0.
    /// </summary>
    /// <typeparam name="TId">The document id type.</typeparam>
    /// <param name="ranked">The ranked list of retrieved document ids (best first).</param>
    /// <param name="relevant">The set of relevant document ids.</param>
    /// <param name="k">The rank cutoff (must be positive).</param>
    /// <returns>1.0 on a hit within the top-k; otherwise 0.0.</returns>
    public static double HitRateAtK<TId>(IReadOnlyList<TId> ranked, ISet<TId> relevant, int k)
    {
        if (ranked == null || relevant == null || relevant.Count == 0 || k <= 0)
            return 0.0;

        int limit = Math.Min(k, ranked.Count);
        for (int i = 0; i < limit; i++)
        {
            if (relevant.Contains(ranked[i]))
                return 1.0;
        }

        return 0.0;
    }

    /// <summary>
    /// Reciprocal rank for a single query: 1 / (rank of the first relevant item), or 0 if none.
    /// </summary>
    /// <typeparam name="TId">The document id type.</typeparam>
    /// <param name="ranked">The ranked list of retrieved document ids (best first).</param>
    /// <param name="relevant">The set of relevant document ids.</param>
    /// <returns>The reciprocal rank in [0, 1].</returns>
    public static double ReciprocalRank<TId>(IReadOnlyList<TId> ranked, ISet<TId> relevant)
    {
        if (ranked == null || relevant == null || relevant.Count == 0)
            return 0.0;

        for (int i = 0; i < ranked.Count; i++)
        {
            if (relevant.Contains(ranked[i]))
                return 1.0 / (i + 1);
        }

        return 0.0;
    }

    /// <summary>
    /// Mean reciprocal rank (MRR) across a set of queries.
    /// </summary>
    /// <typeparam name="TId">The document id type.</typeparam>
    /// <param name="rankings">One ranked list per query.</param>
    /// <param name="relevantSets">One relevant-id set per query, aligned with <paramref name="rankings"/>.</param>
    /// <returns>The mean of the per-query reciprocal ranks.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either sequence is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the two sequences differ in length.</exception>
    public static double MeanReciprocalRank<TId>(
        IEnumerable<IReadOnlyList<TId>> rankings,
        IEnumerable<ISet<TId>> relevantSets)
    {
        var pairs = ZipQueries(rankings, relevantSets);
        if (pairs.Count == 0)
            return 0.0;

        return pairs.Average(p => ReciprocalRank(p.Ranked, p.Relevant));
    }

    /// <summary>
    /// Average precision (AP) for a single query: the mean of Precision@k taken at every rank that
    /// holds a relevant item, divided by the total number of relevant items.
    /// </summary>
    /// <typeparam name="TId">The document id type.</typeparam>
    /// <param name="ranked">The ranked list of retrieved document ids (best first).</param>
    /// <param name="relevant">The set of relevant document ids.</param>
    /// <returns>The average precision in [0, 1]; 0 when there are no relevant items.</returns>
    public static double AveragePrecision<TId>(IReadOnlyList<TId> ranked, ISet<TId> relevant)
    {
        if (ranked == null || relevant == null || relevant.Count == 0)
            return 0.0;

        int hits = 0;
        double sum = 0.0;
        for (int i = 0; i < ranked.Count; i++)
        {
            if (relevant.Contains(ranked[i]))
            {
                hits++;
                sum += (double)hits / (i + 1);
            }
        }

        return sum / relevant.Count;
    }

    /// <summary>
    /// Mean average precision (MAP) across a set of queries.
    /// </summary>
    /// <typeparam name="TId">The document id type.</typeparam>
    /// <param name="rankings">One ranked list per query.</param>
    /// <param name="relevantSets">One relevant-id set per query, aligned with <paramref name="rankings"/>.</param>
    /// <returns>The mean of the per-query average precisions.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either sequence is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the two sequences differ in length.</exception>
    public static double MeanAveragePrecision<TId>(
        IEnumerable<IReadOnlyList<TId>> rankings,
        IEnumerable<ISet<TId>> relevantSets)
    {
        var pairs = ZipQueries(rankings, relevantSets);
        if (pairs.Count == 0)
            return 0.0;

        return pairs.Average(p => AveragePrecision(p.Ranked, p.Relevant));
    }

    /// <summary>
    /// Discounted cumulative gain at rank k with binary relevance and log2 discounting.
    /// </summary>
    /// <typeparam name="TId">The document id type.</typeparam>
    /// <param name="ranked">The ranked list of retrieved document ids (best first).</param>
    /// <param name="relevant">The set of relevant document ids.</param>
    /// <param name="k">The rank cutoff (must be positive).</param>
    /// <returns>DCG@k.</returns>
    public static double DcgAtK<TId>(IReadOnlyList<TId> ranked, ISet<TId> relevant, int k)
    {
        if (ranked == null || relevant == null || k <= 0)
            return 0.0;

        int limit = Math.Min(k, ranked.Count);
        double dcg = 0.0;
        for (int i = 0; i < limit; i++)
        {
            if (relevant.Contains(ranked[i]))
                dcg += 1.0 / Log2(i + 2); // rank position (i+1); discount log2(rank+1) = log2(i+2)
        }

        return dcg;
    }

    /// <summary>
    /// Normalized discounted cumulative gain at rank k with binary relevance.
    /// </summary>
    /// <typeparam name="TId">The document id type.</typeparam>
    /// <param name="ranked">The ranked list of retrieved document ids (best first).</param>
    /// <param name="relevant">The set of relevant document ids.</param>
    /// <param name="k">The rank cutoff (must be positive).</param>
    /// <returns>nDCG@k in [0, 1]; 0 when there are no relevant items.</returns>
    public static double NdcgAtK<TId>(IReadOnlyList<TId> ranked, ISet<TId> relevant, int k)
    {
        if (ranked == null || relevant == null || relevant.Count == 0 || k <= 0)
            return 0.0;

        double dcg = DcgAtK(ranked, relevant, k);

        // Ideal DCG: all relevant items ranked first, up to k positions.
        int idealCount = Math.Min(k, relevant.Count);
        double idcg = 0.0;
        for (int i = 0; i < idealCount; i++)
            idcg += 1.0 / Log2(i + 2);

        return idcg > 0.0 ? dcg / idcg : 0.0;
    }

    /// <summary>
    /// Normalized discounted cumulative gain at rank k with graded (non-binary) relevance gains.
    /// </summary>
    /// <typeparam name="TId">The document id type.</typeparam>
    /// <param name="ranked">The ranked list of retrieved document ids (best first).</param>
    /// <param name="gains">A map from document id to its graded relevance gain (missing ids score 0).</param>
    /// <param name="k">The rank cutoff (must be positive).</param>
    /// <returns>Graded nDCG@k in [0, 1]; 0 when the ideal DCG is 0.</returns>
    public static double NdcgAtK<TId>(IReadOnlyList<TId> ranked, IReadOnlyDictionary<TId, double> gains, int k)
    {
        if (ranked == null || gains == null || k <= 0)
            return 0.0;

        int limit = Math.Min(k, ranked.Count);
        double dcg = 0.0;
        for (int i = 0; i < limit; i++)
        {
            double gain;
            if (gains.TryGetValue(ranked[i], out gain))
                dcg += gain / Log2(i + 2);
        }

        var idealGains = gains.Values.Where(g => g > 0.0).OrderByDescending(g => g).ToList();
        int idealCount = Math.Min(k, idealGains.Count);
        double idcg = 0.0;
        for (int i = 0; i < idealCount; i++)
            idcg += idealGains[i] / Log2(i + 2);

        return idcg > 0.0 ? dcg / idcg : 0.0;
    }

    private static double Log2(double value)
    {
        // Math.Log2 is unavailable on net471; use the change-of-base identity.
        return Math.Log(value) / Math.Log(2.0);
    }

    private static List<(IReadOnlyList<TId> Ranked, ISet<TId> Relevant)> ZipQueries<TId>(
        IEnumerable<IReadOnlyList<TId>> rankings,
        IEnumerable<ISet<TId>> relevantSets)
    {
        if (rankings == null)
            throw new ArgumentNullException(nameof(rankings));
        if (relevantSets == null)
            throw new ArgumentNullException(nameof(relevantSets));

        var rankedList = rankings.ToList();
        var relevantList = relevantSets.ToList();
        if (rankedList.Count != relevantList.Count)
            throw new ArgumentException("The number of rankings must match the number of relevant-id sets.", nameof(relevantSets));

        var pairs = new List<(IReadOnlyList<TId> Ranked, ISet<TId> Relevant)>(rankedList.Count);
        for (int i = 0; i < rankedList.Count; i++)
            pairs.Add((rankedList[i], relevantList[i]));

        return pairs;
    }
}
