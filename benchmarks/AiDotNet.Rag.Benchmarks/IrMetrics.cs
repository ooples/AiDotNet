using System;
using System.Collections.Generic;

namespace AiDotNet.Rag.Benchmarks
{
    /// <summary>
    /// Information-retrieval quality metrics computed from a ranked result list
    /// against a set of known-relevant ids.
    /// </summary>
    internal static class IrMetrics
    {
        /// <summary>
        /// Recall@k = (relevant ids appearing in the top-k) / (total relevant ids).
        /// </summary>
        internal static double RecallAtK(IReadOnlyList<string> ranked, ISet<string> relevant, int k)
        {
            if (relevant.Count == 0)
                return 0.0;

            int hits = 0;
            int limit = Math.Min(k, ranked.Count);
            for (int i = 0; i < limit; i++)
            {
                if (relevant.Contains(ranked[i]))
                    hits++;
            }
            return (double)hits / relevant.Count;
        }

        /// <summary>
        /// Reciprocal rank = 1 / (1-based rank of the first relevant id), or 0 if none.
        /// Averaging this across queries yields MRR.
        /// </summary>
        internal static double ReciprocalRank(IReadOnlyList<string> ranked, ISet<string> relevant)
        {
            for (int i = 0; i < ranked.Count; i++)
            {
                if (relevant.Contains(ranked[i]))
                    return 1.0 / (i + 1);
            }
            return 0.0;
        }

        /// <summary>
        /// Normalized discounted cumulative gain over the top-k with binary relevance.
        /// DCG = sum_i rel_i / log2(i + 2); IDCG is the DCG of the ideal ranking.
        /// </summary>
        internal static double NdcgAtK(IReadOnlyList<string> ranked, ISet<string> relevant, int k)
        {
            if (relevant.Count == 0)
                return 0.0;

            double dcg = 0.0;
            int limit = Math.Min(k, ranked.Count);
            for (int i = 0; i < limit; i++)
            {
                if (relevant.Contains(ranked[i]))
                    dcg += 1.0 / Math.Log2(i + 2);
            }

            // Ideal DCG: all relevant docs packed at the front, capped at k.
            double idcg = 0.0;
            int ideal = Math.Min(relevant.Count, k);
            for (int i = 0; i < ideal; i++)
                idcg += 1.0 / Math.Log2(i + 2);

            return idcg == 0.0 ? 0.0 : dcg / idcg;
        }

        /// <summary>
        /// Recall@k for an ANN index measured against a brute-force ground-truth
        /// neighbor set (both are top-k id lists of the same length).
        /// </summary>
        internal static double NeighborRecall(IReadOnlyList<string> approx, ISet<string> groundTruth, int k)
        {
            if (groundTruth.Count == 0)
                return 0.0;

            int hits = 0;
            int limit = Math.Min(k, approx.Count);
            for (int i = 0; i < limit; i++)
            {
                if (groundTruth.Contains(approx[i]))
                    hits++;
            }
            return (double)hits / Math.Min(k, groundTruth.Count);
        }
    }

    /// <summary>Percentile helpers over a mutable list of latency samples.</summary>
    internal static class Stats
    {
        /// <summary>Nearest-rank percentile. <paramref name="samples"/> is sorted in place.</summary>
        internal static double Percentile(List<double> samples, double p)
        {
            if (samples.Count == 0)
                return 0.0;
            samples.Sort();
            int rank = (int)Math.Ceiling(p / 100.0 * samples.Count) - 1;
            if (rank < 0) rank = 0;
            if (rank >= samples.Count) rank = samples.Count - 1;
            return samples[rank];
        }
    }
}
