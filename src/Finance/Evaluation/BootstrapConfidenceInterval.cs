using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Finance.Evaluation;

/// <summary>
/// Bootstrap confidence intervals for a statistic of a return series. Supports the IID bootstrap and the
/// stationary (Politis-Romano) block bootstrap, which preserves serial dependence in time-series returns.
/// Built-in statistics include the mean and the (per-observation) Sharpe ratio; any custom statistic can
/// be supplied.
/// </summary>
/// <remarks>
/// <para>
/// A point estimate (e.g. a backtest's Sharpe ratio) is just one number; the bootstrap quantifies how much
/// it would wobble under resampling, yielding a percentile confidence interval without distributional
/// assumptions. The stationary bootstrap resamples random-length blocks (geometric length distribution)
/// rather than single observations, so autocorrelation in returns is not destroyed.
/// </para>
/// <para><b>For Beginners:</b> Suppose your strategy's Sharpe ratio is 1.2. Is that "1.2 give or take 0.1",
/// or "1.2 give or take 0.8"? The bootstrap answers this by repeatedly re-sampling your return history
/// (with replacement) thousands of times, re-computing the statistic each time, and looking at the spread.
/// The middle 95% of those values is your confidence interval. We resample whole <i>blocks</i> of returns
/// (not isolated days) so that streaks and momentum in the data are respected. A seeded random generator
/// is passed in so the result is exactly reproducible.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float/double).</typeparam>
public static class BootstrapConfidenceInterval<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The percentile confidence interval plus the point estimate computed on the full sample.</summary>
    public sealed class Interval
    {
        /// <summary>The statistic evaluated on the original (un-resampled) sample.</summary>
        public T PointEstimate { get; }

        /// <summary>Lower percentile bound of the bootstrap distribution.</summary>
        public T Lower { get; }

        /// <summary>Upper percentile bound of the bootstrap distribution.</summary>
        public T Upper { get; }

        /// <summary>Creates an interval result.</summary>
        public Interval(T pointEstimate, T lower, T upper)
        {
            PointEstimate = pointEstimate;
            Lower = lower;
            Upper = upper;
        }
    }

    /// <summary>Mean of a return sample (per-observation).</summary>
    public static double MeanStatistic(IReadOnlyList<double> sample)
    {
        if (sample.Count == 0)
        {
            return 0.0;
        }

        double s = 0.0;
        for (int i = 0; i < sample.Count; i++)
        {
            s += sample[i];
        }

        return s / sample.Count;
    }

    /// <summary>Per-observation Sharpe ratio (mean / sample-stdev) of a return sample.</summary>
    public static double SharpeStatistic(IReadOnlyList<double> sample)
    {
        int n = sample.Count;
        if (n < 2)
        {
            return 0.0;
        }

        double mean = MeanStatistic(sample);
        double sse = 0.0;
        for (int i = 0; i < n; i++)
        {
            double d = sample[i] - mean;
            sse += d * d;
        }

        double std = Math.Sqrt(sse / (n - 1));
        return std > 0.0 ? mean / std : 0.0;
    }

    /// <summary>
    /// Computes a bootstrap percentile confidence interval.
    /// </summary>
    /// <param name="returns">The return series (time-ordered for the stationary bootstrap).</param>
    /// <param name="rng">A seeded <see cref="Random"/> for reproducibility (no global RNG dependence).</param>
    /// <param name="statistic">
    /// The statistic to bootstrap, as a function of a resampled series. Defaults to the Sharpe ratio.
    /// </param>
    /// <param name="confidence">Confidence level in (0, 1), e.g. 0.95 for a 95% CI.</param>
    /// <param name="nResamples">Number of bootstrap resamples.</param>
    /// <param name="meanBlockLength">
    /// Mean block length for the stationary bootstrap (geometric distribution). Use 1 for the plain IID
    /// bootstrap. Larger values preserve more serial dependence.
    /// </param>
    /// <returns>The point estimate and the lower/upper percentile bounds.</returns>
    public static Interval Compute(
        IReadOnlyList<T> returns,
        Random rng,
        Func<IReadOnlyList<double>, double>? statistic = null,
        double confidence = 0.95,
        int nResamples = 1000,
        double meanBlockLength = 1.0)
    {
        if (rng == null)
        {
            throw new ArgumentNullException(nameof(rng));
        }

        if (confidence <= 0.0 || confidence >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(confidence), "confidence must be in (0, 1).");
        }

        if (nResamples < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nResamples), "nResamples must be at least 1.");
        }

        if (meanBlockLength < 1.0)
        {
            meanBlockLength = 1.0;
        }

        var stat = statistic ?? SharpeStatistic;

        int n = returns.Count;
        var data = new double[n];
        for (int i = 0; i < n; i++)
        {
            data[i] = NumOps.ToDouble(returns[i]);
        }

        double pointEstimate = stat(data);

        if (n < 2)
        {
            var pe = NumOps.FromDouble(pointEstimate);
            return new Interval(pe, pe, pe);
        }

        // Probability of starting a new block each step (stationary bootstrap): p = 1 / meanBlockLength.
        double pNewBlock = 1.0 / meanBlockLength;

        var estimates = new double[nResamples];
        var resample = new double[n];
        for (int b = 0; b < nResamples; b++)
        {
            int idx = rng.Next(n);
            for (int i = 0; i < n; i++)
            {
                if (i == 0 || rng.NextDouble() < pNewBlock)
                {
                    idx = rng.Next(n);
                }
                else
                {
                    idx++;
                    if (idx >= n)
                    {
                        idx = 0; // wrap (circular) — stationary bootstrap.
                    }
                }

                resample[i] = data[idx];
            }

            estimates[b] = stat(resample);
        }

        Array.Sort(estimates);
        double alpha = 1.0 - confidence;
        double lower = Percentile(estimates, alpha / 2.0 * 100.0);
        double upper = Percentile(estimates, (1.0 - alpha / 2.0) * 100.0);

        return new Interval(
            NumOps.FromDouble(pointEstimate),
            NumOps.FromDouble(lower),
            NumOps.FromDouble(upper));
    }

    /// <summary>Linear-interpolated percentile of a pre-sorted array (percentile in [0, 100]).</summary>
    private static double Percentile(double[] sorted, double percentile)
    {
        int n = sorted.Length;
        if (n == 1)
        {
            return sorted[0];
        }

        double rank = percentile / 100.0 * (n - 1);
        int lo = (int)Math.Floor(rank);
        int hi = (int)Math.Ceiling(rank);
        if (lo < 0)
        {
            lo = 0;
        }

        if (hi >= n)
        {
            hi = n - 1;
        }

        if (lo == hi)
        {
            return sorted[lo];
        }

        double frac = rank - lo;
        return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
    }
}
