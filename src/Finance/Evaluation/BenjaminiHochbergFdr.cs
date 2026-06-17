using System;
using System.Collections.Generic;

namespace AiDotNet.Finance.Evaluation;

/// <summary>
/// Benjamini-Hochberg false-discovery-rate (FDR) control for multiple hypothesis testing. Given a set of
/// p-values and a target FDR level alpha, it decides which hypotheses to reject and returns BH-adjusted
/// q-values.
/// </summary>
/// <remarks>
/// <para>
/// When many strategies / signals / features are each tested for significance, the chance of at least one
/// false positive explodes. The Benjamini-Hochberg procedure controls the expected proportion of false
/// discoveries among the rejected hypotheses at level alpha: sort the p-values ascending, find the largest
/// rank k with p_(k) ≤ (k / m)·alpha, and reject all hypotheses with rank ≤ k. The q-value is the
/// monotone-adjusted p-value (min over later ranks of m/rank · p), capped at 1.
/// </para>
/// <para><b>For Beginners:</b> Say you backtest 100 signals and 5 look "significant" at the usual p &lt; 0.05.
/// But testing 100 things means several will look good by pure chance. BH-FDR is a smarter, less strict
/// cousin of the Bonferroni correction: instead of demanding every signal clear an impossibly high bar, it
/// keeps the <i>fraction</i> of bogus discoveries among your accepted signals below a level you choose
/// (say 10%). The q-value is "the smallest FDR at which this hypothesis would still be accepted" — handy
/// for ranking discoveries.</para>
/// </remarks>
public static class BenjaminiHochbergFdr
{
    /// <summary>Result of the BH procedure: per-hypothesis rejection flags and adjusted q-values.</summary>
    public sealed class Result
    {
        /// <summary>Rejected[i] is true when hypothesis i is rejected (a discovery) at the given alpha.</summary>
        public IReadOnlyList<bool> Rejected { get; }

        /// <summary>BH-adjusted q-values, aligned to the input p-value order, each in [0, 1].</summary>
        public IReadOnlyList<double> QValues { get; }

        /// <summary>Number of rejected hypotheses (discoveries).</summary>
        public int NumRejected { get; }

        /// <summary>Creates a BH result.</summary>
        public Result(IReadOnlyList<bool> rejected, IReadOnlyList<double> qValues, int numRejected)
        {
            Rejected = rejected;
            QValues = qValues;
            NumRejected = numRejected;
        }
    }

    /// <summary>
    /// Runs the Benjamini-Hochberg FDR procedure.
    /// </summary>
    /// <param name="pValues">The p-values, one per hypothesis (each in [0, 1]).</param>
    /// <param name="alpha">Target false discovery rate, in (0, 1). Defaults to 0.05.</param>
    /// <returns>Rejection flags (input order), adjusted q-values (input order), and the rejection count.</returns>
    public static Result Apply(IReadOnlyList<double> pValues, double alpha = 0.05)
    {
        if (pValues == null)
        {
            throw new ArgumentNullException(nameof(pValues));
        }

        if (alpha <= 0.0 || alpha >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha), "alpha must be in (0, 1).");
        }

        int m = pValues.Count;
        var rejected = new bool[m];
        var qValues = new double[m];
        if (m == 0)
        {
            return new Result(rejected, qValues, 0);
        }

        // Sort hypothesis indices by ascending p-value.
        var order = new int[m];
        for (int i = 0; i < m; i++)
        {
            order[i] = i;
        }

        Array.Sort(order, (a, b) => pValues[a].CompareTo(pValues[b]));

        // Largest rank k (1-based) with p_(k) <= (k/m)*alpha => reject ranks 1..k.
        int maxRejectRank = 0;
        for (int rank = 1; rank <= m; rank++)
        {
            double p = pValues[order[rank - 1]];
            if (p <= (double)rank / m * alpha)
            {
                maxRejectRank = rank;
            }
        }

        for (int rank = 1; rank <= maxRejectRank; rank++)
        {
            rejected[order[rank - 1]] = true;
        }

        // Adjusted q-values: enforce monotonicity from the largest rank downward.
        // q_(k) = min over j>=k of ( m/j * p_(j) ), capped at 1.
        double running = 1.0;
        for (int rank = m; rank >= 1; rank--)
        {
            double p = pValues[order[rank - 1]];
            double q = (double)m / rank * p;
            if (q < running)
            {
                running = q;
            }

            double capped = running < 1.0 ? running : 1.0;
            qValues[order[rank - 1]] = capped;
        }

        return new Result(rejected, qValues, maxRejectRank);
    }
}
