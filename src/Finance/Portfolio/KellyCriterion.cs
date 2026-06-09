using System;
using System.Collections.Generic;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Finance.Portfolio;

/// <summary>
/// Kelly-criterion position sizing — the bet fraction that maximizes long-run log-growth of capital.
/// </summary>
/// <remarks>
/// <para>
/// AiDotNet's portfolio optimizers cover weight allocation (mean-variance, HRP, Black-Litterman) but
/// not Kelly bet sizing. This fills that gap with the two standard forms — discrete (win probability +
/// payoff odds) and continuous (mean / variance of returns) — plus fractional Kelly, which is what
/// practitioners actually use because full Kelly is famously over-aggressive.
/// </para>
/// <para><b>For Beginners:</b> Kelly tells you what fraction of your capital to put on a trade to grow
/// wealth fastest over many trades. Bet more than Kelly and you risk ruin; bet a fraction of it
/// ("half-Kelly") for much lower volatility at a small growth cost. A negative Kelly means "no edge —
/// don't take the trade."</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float/double).</typeparam>
/// <remarks>
/// Implements <see cref="IPositionSizer{T}"/> so it can be injected as the default, swappable sizing
/// strategy into trading/portfolio models; the static methods remain available for direct use.
/// </remarks>
public class KellyCriterion<T> : IPositionSizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Shared stateless default instance for injection as an <see cref="IPositionSizer{T}"/>.</summary>
    public static IPositionSizer<T> Default { get; } = new KellyCriterion<T>();

    // Explicit IPositionSizer<T> members delegate to the static implementations.
    T IPositionSizer<T>.Discrete(T winProbability, T winLossRatio) => Discrete(winProbability, winLossRatio);
    T IPositionSizer<T>.Continuous(T expectedReturn, T variance) => Continuous(expectedReturn, variance);
    T IPositionSizer<T>.Fractional(T baseFraction, double fraction) => Fractional(baseFraction, fraction);
    T IPositionSizer<T>.FromReturns(IEnumerable<T> returns, double fraction) => FromReturns(returns, fraction);

    /// <summary>
    /// Discrete Kelly fraction: f* = p − (1 − p) / b, where <paramref name="winProbability"/> is p and
    /// <paramref name="winLossRatio"/> is b (net payoff on a win per unit risked on a loss). Returns 0
    /// when there is no edge (negative) or the inputs are degenerate.
    /// </summary>
    public static T Discrete(T winProbability, T winLossRatio)
    {
        // winProbability ∈ [0, 1]: reject out-of-range probabilities (expressed via
        // GreaterThan/LessThan so it works for every INumericOperations<T>).
        if (NumOps.LessThan(winProbability, NumOps.Zero) || NumOps.GreaterThan(winProbability, NumOps.One))
        {
            throw new ArgumentOutOfRangeException(nameof(winProbability), "winProbability must be in [0, 1].");
        }

        if (!NumOps.GreaterThan(winLossRatio, NumOps.Zero))
        {
            return NumOps.Zero;
        }

        var lossProbability = NumOps.Subtract(NumOps.One, winProbability);
        var kelly = NumOps.Subtract(winProbability, NumOps.Divide(lossProbability, winLossRatio));
        return NumOps.GreaterThan(kelly, NumOps.Zero) ? kelly : NumOps.Zero;
    }

    /// <summary>
    /// Continuous Kelly fraction for (approximately) Gaussian returns: f* = μ / σ², where
    /// <paramref name="expectedReturn"/> is μ and <paramref name="variance"/> is σ². This is the
    /// vol-targeted exposure that maximizes expected log-growth. Returns 0 for non-positive variance.
    /// </summary>
    public static T Continuous(T expectedReturn, T variance)
    {
        return NumOps.GreaterThan(variance, NumOps.Zero)
            ? NumOps.Divide(expectedReturn, variance)
            : NumOps.Zero;
    }

    /// <summary>
    /// Fractional Kelly: <paramref name="kellyFraction"/> scaled by <paramref name="fraction"/> (e.g.
    /// 0.5 for half-Kelly). The standard risk-managed form — much lower drawdown for a small growth cost.
    /// </summary>
    public static T Fractional(T kellyFraction, double fraction)
        => NumOps.Multiply(kellyFraction, NumOps.FromDouble(fraction));

    /// <summary>
    /// Continuous Kelly estimated from a return series: computes the sample mean and (population)
    /// variance of <paramref name="returns"/> and applies <see cref="Continuous"/>. Optionally scaled by
    /// <paramref name="fraction"/> (default 1.0 = full Kelly).
    /// </summary>
    public static T FromReturns(IEnumerable<T> returns, double fraction = 1.0)
    {
        if (returns is null)
        {
            throw new ArgumentNullException(nameof(returns));
        }

        // Materialize once: the source may be a deferred / non-repeatable
        // enumerable (a LINQ query, a generator), so enumerating it twice could
        // recompute or yield nothing the second pass.
        var series = returns as IReadOnlyList<T> ?? new List<T>(returns);

        var count = series.Count;
        if (count < 2)
        {
            return NumOps.Zero;
        }

        var sum = NumOps.Zero;
        foreach (var r in series)
        {
            sum = NumOps.Add(sum, r);
        }

        var mean = NumOps.Divide(sum, NumOps.FromDouble(count));
        var sse = NumOps.Zero;
        foreach (var r in series)
        {
            var d = NumOps.Subtract(r, mean);
            sse = NumOps.Add(sse, NumOps.Multiply(d, d));
        }

        var variance = NumOps.Divide(sse, NumOps.FromDouble(count));
        return Fractional(Continuous(mean, variance), fraction);
    }
}
